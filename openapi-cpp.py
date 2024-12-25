#!/usr/bin/env python3

import sys
import yaml

# Map (type, format) -> C++ type for simple scalars
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
        """
        doc: The parsed OpenAPI spec (a Python dict).
        """
        self.doc = doc
        self.components = doc.get('components', {})
        self.schemas = self.components.get('schemas', {})
        # Keep track of schemas we've generated so we don't do it multiple times
        self.generated = set()
        # We'll store forward declarations and struct definitions separately
        self.forward_decls = []
        self.struct_defs = []

    def ref_to_name(self, ref_str):
        """
        Converts something like '#/components/schemas/Foo' -> 'Foo'
        """
        return ref_str.split('/')[-1]

    def get_cpp_type(self, prop_schema):
        """
        Returns a C++ type for this property, or UNKNOWN_TYPE if we can't map it.
        We handle:
          - $ref
          - scalar types
          - arrays
          - 'object' (inline) -> UNKNOWN_TYPE by default
          - 'enum' -> std::string (placeholder)
        """
        if not isinstance(prop_schema, dict):
            return "UNKNOWN_TYPE"

        # If there's a reference, we eventually want the struct name
        if '$ref' in prop_schema:
            return self.ref_to_name(prop_schema['$ref'])

        prop_type = prop_schema.get('type')
        prop_format = prop_schema.get('format')

        # arrays
        if prop_type == 'array':
            items = prop_schema.get('items', {})
            elem_type = self.get_cpp_type(items)
            return f"std::vector<{elem_type}>"

        # object (inline) -> we do a fallback
        if prop_type == 'object':
            return "UNKNOWN_TYPE"

        # enum
        if 'enum' in prop_schema:
            # real code might parse out the enum values
            return "std::string  // or generate an enum"

        # fallback to scalar mapping
        return BASIC_TYPE_MAP.get((prop_type, prop_format), "UNKNOWN_TYPE")

    def generate_schema(self, schema_name):
        """
        Generate a forward declaration and a struct for the given schema.
        - If the schema has references, we ensure those schemas are generated first.
        - Then we produce a single struct with the discovered properties as members.
        """
        if schema_name in self.generated:
            return  # already processed

        schema = self.schemas.get(schema_name)
        self.generated.add(schema_name)

        # If we don't have a real schema dict, just forward-declare
        if not schema or not isinstance(schema, dict):
            self.forward_decls.append(f"struct {schema_name};")
            return

        # Forward declare
        self.forward_decls.append(f"struct {schema_name};")

        # Collect the properties
        props = schema.get('properties', {})
        required = schema.get('required', [])

        struct_lines = []
        struct_lines.append(f"struct {schema_name} {{")

        for prop_name, prop_schema in props.items():
            # If $ref, we must generate the referenced schema first
            if '$ref' in prop_schema:
                ref_name = self.ref_to_name(prop_schema['$ref'])
                self.generate_schema(ref_name)

            cpp_type = self.get_cpp_type(prop_schema)
            # If not in required, you might want to wrap in std::optional
            # if prop_name not in required:
            #     cpp_type = f"std::optional<{cpp_type}>"

            struct_lines.append(f"    {cpp_type} {prop_name};")

        struct_lines.append("};\n")
        self.struct_defs.append("\n".join(struct_lines))

    def generate_all(self):
        """
        Generate code for every schema in components.schemas.
        """
        for name in sorted(self.schemas.keys()):
            self.generate_schema(name)

    def produce_code(self):
        """
        Produce the final combined C++ header as a string.
        """
        lines = []
        lines.append("#pragma once")
        lines.append("#include <string>")
        lines.append("#include <vector>")
        lines.append("#include <optional>")
        lines.append("")
        lines.append("// If the generator can't map a type, we use UNKNOWN_TYPE to force a compile error.")
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
