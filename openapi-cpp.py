#!/usr/bin/env python3

import sys
import yaml

BASIC_TYPE_MAP = {
    ('string', None): 'std::string',
    ('integer', None): 'int',
    ('integer', 'int32'): 'int',
    ('integer', 'int64'): 'long long',
    ('number', None): 'double',
    ('boolean', None): 'bool'
}

class NormalStructGenerator:
    def __init__(self, doc):
        self.doc = doc
        self.components = doc.get('components', {})
        self.schemas = self.components.get('schemas', {})
        self.generated = set()
        self.forward_decls = []
        self.struct_defs = []

    def ref_to_name(self, ref_str):
        return ref_str.split('/')[-1]

    def get_cpp_type_and_comment(self, prop_schema):
        """
        Return (cpp_type, comment).
        The comment will be appended after the semicolon, e.g.:
           <cpp_type> <prop_name>; // <comment>
        """
        if not isinstance(prop_schema, dict):
            return ("UNKNOWN_TYPE", "")

        # If $ref
        if '$ref' in prop_schema:
            return (self.ref_to_name(prop_schema['$ref']), "")

        prop_type = prop_schema.get('type')
        prop_format = prop_schema.get('format')

        # arrays
        if prop_type == 'array':
            items = prop_schema.get('items', {})
            elem_type, elem_comment = self.get_cpp_type_and_comment(items)
            # For simplicity, ignore element comment. Or you can incorporate it:
            return (f"std::vector<{elem_type}>", elem_comment)

        # inline object
        if prop_type == 'object':
            return ("UNKNOWN_TYPE", "")

        # enum
        if 'enum' in prop_schema:
            return ("std::string", "or generate an enum type")

        # fallback scalar
        mapped = BASIC_TYPE_MAP.get((prop_type, prop_format), None)
        if mapped:
            return (mapped, "")
        else:
            return ("UNKNOWN_TYPE", "")

    def generate_schema(self, schema_name):
        if schema_name in self.generated:
            return

        schema = self.schemas.get(schema_name)
        self.generated.add(schema_name)

        if not schema or not isinstance(schema, dict):
            self.forward_decls.append(f"struct {schema_name};")
            return

        self.forward_decls.append(f"struct {schema_name};")

        props = schema.get('properties', {})
        required = schema.get('required', [])

        struct_lines = []
        struct_lines.append(f"struct {schema_name} {{")

        for prop_name, prop_schema in props.items():
            # If referencing another schema
            if '$ref' in prop_schema:
                ref_name = self.ref_to_name(prop_schema['$ref'])
                self.generate_schema(ref_name)

            cpp_type, comment = self.get_cpp_type_and_comment(prop_schema)

            # Optionally do: if prop_name not in required => std::optional
            # if prop_name not in required:
            #     cpp_type = f"std::optional<{cpp_type}>"

            # If there's a comment, we append it after the semicolon.
            if comment:
                struct_lines.append(f"    {cpp_type} {prop_name}; // {comment}")
            else:
                struct_lines.append(f"    {cpp_type} {prop_name};")

        struct_lines.append("};\n")
        self.struct_defs.append("\n".join(struct_lines))

    def generate_all(self):
        for name in sorted(self.schemas.keys()):
            self.generate_schema(name)

    def produce_code(self):
        lines = []
        lines.append("#pragma once")
        lines.append("#include <string>")
        lines.append("#include <vector>")
        lines.append("#include <optional>")
        lines.append("")
        lines.append('// If we can\'t map something, use UNKNOWN_TYPE => compile error.')
        lines.append('#define UNKNOWN_TYPE static_assert(false, "Unknown schema type encountered!");')
        lines.append("")
        lines.append("// Forward declarations:")
        lines.extend(self.forward_decls)
        lines.append("\n// Struct definitions:")
        lines.extend(self.struct_defs)
        return "\n".join(lines)

def main(openapi_path, output_path):
    with open(openapi_path, 'r') as f:
        doc = yaml.safe_load(f)

    generator = NormalStructGenerator(doc)
    generator.generate_all()
    code = generator.produce_code()

    if output_path == '-':
        print(code)
    else:
        with open(output_path, 'w') as out:
            out.write(code)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <openapi.yaml> [output_file or -]")
        sys.exit(1)

    openapi_file = sys.argv[1]
    if len(sys.argv) > 2:
        out_file = sys.argv[2]
    else:
        out_file = '-'
    main(openapi_file, out_file)
