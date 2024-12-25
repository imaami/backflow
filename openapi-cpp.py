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

class CRTPGeneratorRefsAsMembers:
    def __init__(self, doc):
        """
        doc: Parsed OpenAPI or JSON Schema that has a 'components'/'schemas' block.
        """
        self.doc = doc
        self.components = doc.get('components', {})
        self.schemas = self.components.get('schemas', {})
        # Keep track of which schemas we've generated
        self.generated = set()
        # We'll store lines of code in these lists
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
        Return a C++ type for a property that is not itself a schema-level reference.

        - If 'type=array', produce std::vector<...>.
        - If 'type=object' with no $ref, treat it as UNKNOWN_TYPE (or generate an inline struct, if you prefer).
        - If there's an enum, produce std::string (for now).
        - Otherwise, use BASIC_TYPE_MAP or fallback to UNKNOWN_TYPE.
        """
        if not isinstance(prop_schema, dict):
            return "UNKNOWN_TYPE"  # not a recognized dictionary

        if '$ref' in prop_schema:
            # We'll handle references by returning the final derived struct's name
            return self.ref_to_name(prop_schema['$ref'])

        prop_type = prop_schema.get('type')
        prop_format = prop_schema.get('format')

        if prop_type == 'array':
            items = prop_schema.get('items', {})
            elem_type = self.get_cpp_type(items)
            return f"std::vector<{elem_type}>"

        if prop_type == 'object':
            # For inline objects, we either generate a new inline struct or skip.
            # We'll do a fallback to UNKNOWN_TYPE for simplicity.
            return "UNKNOWN_TYPE"

        if 'enum' in prop_schema:
            # Could generate an enum, but here we store as string
            return "std::string  // or generated enum"

        # If it's a scalar type:
        return BASIC_TYPE_MAP.get((prop_type, prop_format), "UNKNOWN_TYPE")

    def generate_schema(self, schema_name):
        """
        Create the CRTP base template and derived struct for schema_name.
        - We do NOT inherit from references. Instead, references become data members.
        """
        if schema_name in self.generated:
            return  # Already done

        schema = self.schemas.get(schema_name)
        if not schema or not isinstance(schema, dict):
            # We have no info about this schema, forward-declare at least
            self.forward_declarations.append(f"struct {schema_name};")
            self.generated.add(schema_name)
            return

        # Mark as generated to avoid re-entrance
        self.generated.add(schema_name)

        # Add forward declarations for the base & derived
        self.forward_declarations.append(f"template <typename Derived> struct {schema_name}Base;")
        self.forward_declarations.append(f"struct {schema_name};")

        # Gather properties
        props = schema.get('properties', {})
        required = schema.get('required', [])

        base_lines = []
        base_lines.append(f"template <typename Derived>")
        base_lines.append(f"struct {schema_name}Base {{")

        # For each property:
        for prop_name, prop_schema in props.items():
            # If this property references another schema, we must generate that schema first
            # (We won't do inheritance, we'll just store that derived struct as a member.)
            if '$ref' in prop_schema:
                ref_name = self.ref_to_name(prop_schema['$ref'])
                # Ensure that referenced schema is generated
                self.generate_schema(ref_name)
                # So the property type is 'ref_name' (the final derived struct)
                cpp_type = ref_name
            else:
                # Normal property (scalar, array, object, or fallback)
                cpp_type = self.get_cpp_type(prop_schema)

            # If not required, you might want to wrap in std::optional:
            # if prop_name not in required:
            #     cpp_type = f"std::optional<{cpp_type}>"

            base_lines.append(f"    {cpp_type} {prop_name};")

        base_lines.append("};\n")
        self.base_definitions.append("\n".join(base_lines))

        # Then the derived struct:
        derived_lines = []
        derived_lines.append(f"struct {schema_name} : public {schema_name}Base<{schema_name}> {{")
        derived_lines.append("    // specialized methods if needed")
        derived_lines.append("};\n")
        self.derived_definitions.append("\n".join(derived_lines))

    def generate_all(self):
        """
        Generate code for all schemas.
        """
        for name in sorted(self.schemas.keys()):
            self.generate_schema(name)

    def produce_code(self):
        """
        Produce the final C++ code as a single string.
        """
        output = []
        output.append("#pragma once")
        output.append("#include <string>")
        output.append("#include <vector>")
        output.append("#include <optional>")
        output.append("")
        output.append("// Fallback for unknown or unhandled schema types:")
        output.append('#define UNKNOWN_TYPE static_assert(false, "Unknown schema type encountered!");')
        output.append("")
        output.append("// Forward declarations:")
        output.extend(self.forward_declarations)
        output.append("\n// Base definitions (CRTP):")
        output.extend(self.base_definitions)
        output.append("\n// Derived definitions:")
        output.extend(self.derived_definitions)
        return "\n".join(output)

def main(openapi_path, output_path):
    with open(openapi_path, 'r') as f:
        doc = yaml.safe_load(f)

    generator = CRTPGeneratorRefsAsMembers(doc)
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
