#!/usr/bin/env python3

import sys
import yaml

# Map (type, format) -> C++ type for scalars
BASIC_TYPE_MAP = {
    ('string', None): 'std::string',
    ('integer', None): 'int',
    ('integer', 'int32'): 'int',
    ('integer', 'int64'): 'long long',
    ('number', None): 'double',
    ('boolean', None): 'bool'
}

class CRTPGenerator:
    def __init__(self, doc):
        self.doc = doc
        self.components = doc.get('components', {})
        self.schemas = self.components.get('schemas', {})
        # Keep track of which schemas we've generated
        self.generated = set()
        # We'll store lines of code in lists
        self.forward_declarations = []
        self.base_definitions = []
        self.derived_definitions = []

    def ref_to_name(self, ref_str):
        """
        E.g.: '#/components/schemas/Foo' -> 'Foo'
        """
        return ref_str.split('/')[-1]

    def get_cpp_type(self, prop_schema):
        """
        Return a C++ type or a CRTP base. If $ref, we won't store a direct type for a property,
        because we plan to do inheritance instead. For arrays/scalars, we map them in a naive way.
        """
        if not isinstance(prop_schema, dict):
            # Not a recognized dictionary schema; fallback to an error type
            return "UNKNOWN_TYPE"

        if '$ref' in prop_schema:
            # We'll handle references via inheritance in the parent schema,
            # but let's return the referenced name if needed.
            return self.ref_to_name(prop_schema['$ref'])

        prop_type = prop_schema.get('type')
        prop_format = prop_schema.get('format')
        if prop_type == 'array':
            # We do a naive single-level approach
            items = prop_schema.get('items', {})
            elem_type = self.get_cpp_type(items)
            return f"std::vector<{elem_type}>"

        if prop_type == 'object':
            # Possibly an inline object (no $ref).
            # The fallback is to say "We don't handle it" → UNKNOWN_TYPE
            # Or keep it as "T" if you want a placeholder. We'll treat it as unknown for now:
            return "UNKNOWN_TYPE"

        # If there's an enum, do something naive
        if 'enum' in prop_schema:
            return "std::string  // or generate an enum"

        # Otherwise, a scalar:
        return BASIC_TYPE_MAP.get((prop_type, prop_format), "UNKNOWN_TYPE")

    def generate_schema(self, schema_name):
        """
        Create the CRTP base template and derived struct for a given schema name.
        """
        if schema_name in self.generated:
            return
        self.generated.add(schema_name)

        schema = self.schemas.get(schema_name)
        if not schema or not isinstance(schema, dict):
            # Forward declare and skip
            self.forward_declarations.append(f"struct {schema_name};")
            return

        # Forward declarations
        self.forward_declarations.append(f"template <typename Derived> struct {schema_name}Base;")
        self.forward_declarations.append(f"struct {schema_name};")

        # Identify properties
        props = schema.get('properties', {})
        required = schema.get('required', [])

        # Figure out which references we inherit from
        inherited_refs = []
        normal_props = []

        for prop_name, prop_schema in props.items():
            if '$ref' in prop_schema:
                ref_name = self.ref_to_name(prop_schema['$ref'])
                # Generate that referenced schema first
                self.generate_schema(ref_name)
                inherited_refs.append(ref_name)
            else:
                # It's a normal property
                ctype = self.get_cpp_type(prop_schema)
                # If not required, you might wrap it in std::optional here:
                # if prop_name not in required:
                #     ctype = f"std::optional<{ctype}>"
                normal_props.append((prop_name, ctype))

        # Build the inheritance line:
        # e.g. "struct FooBase : public BarBase<Derived>, public BazBase<Derived>"
        inheritance_list = []
        for ref_ in inherited_refs:
            inheritance_list.append(f"public {ref_}Base<Derived>")
        if inheritance_list:
            inheritance_str = " : " + ", ".join(inheritance_list)
        else:
            inheritance_str = ""

        # Build the CRTP base struct definition
        base_lines = []
        base_lines.append(f"template <typename Derived>")
        base_lines.append(f"struct {schema_name}Base{inheritance_str} {{")

        if normal_props:
            for (prop_name, ctype) in normal_props:
                base_lines.append(f"    {ctype} {prop_name};")
        else:
            base_lines.append("    // no direct properties")

        base_lines.append("};\n")
        self.base_definitions.append("\n".join(base_lines))

        # Build the derived struct
        derived_lines = []
        derived_lines.append(f"struct {schema_name} : public {schema_name}Base<{schema_name}> {{")
        derived_lines.append("    // specialized methods if needed")
        derived_lines.append("};\n")
        self.derived_definitions.append("\n".join(derived_lines))

    def generate_all(self):
        """
        Generate code for all schemas we know about.
        """
        for name in sorted(self.schemas.keys()):
            self.generate_schema(name)

    def produce_code(self):
        code = []
        code.append("#pragma once")
        code.append("#include <string>")
        code.append("#include <vector>")
        code.append("#include <optional>")
        code.append("")
        code.append("// If we ever generate a fallback or unknown type, this macro triggers a compile-time error.")
        code.append('#define UNKNOWN_TYPE static_assert(false, "Unknown schema type encountered!");')
        code.append("")
        code.append("// Forward declarations:")
        code.extend(self.forward_declarations)
        code.append("\n// Base definitions (CRTP):")
        code.extend(self.base_definitions)
        code.append("\n// Derived definitions:")
        code.extend(self.derived_definitions)
        return "\n".join(code)

def main(openapi_path, output_path):
    with open(openapi_path, 'r') as f:
        doc = yaml.safe_load(f)

    generator = CRTPGenerator(doc)
    generator.generate_all()
    code = generator.produce_code()

    if output_path == '-':
        print(code)
    else:
        with open(output_path, 'w') as out:
            out.write(code)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_crtp_code.py <openapi.yaml> [output.cpp or -]")
        sys.exit(1)

    openapi_file = sys.argv[1]
    if len(sys.argv) > 2:
        out_file = sys.argv[2]
    else:
        out_file = '-'
    main(openapi_file, out_file)
