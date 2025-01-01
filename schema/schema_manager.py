from typing import List, Literal, Type, Dict
from pathlib import Path
import importlib.util
import logging
from pydantic import BaseModel, create_model
import inspect


class SchemaManager:
    def __init__(self):
        self.schemas: Dict[str, Type[BaseModel]] = {}
        self._load_schemas()

    def _load_schemas(self) -> None:
        """Scan and load all schema files from the schema directory."""
        schema_dir = Path(__file__).parent
        for schema_file in schema_dir.glob("*.py"):
            if schema_file.stem == "schema_manager":
                continue

            try:
                # Import the module dynamically
                spec = importlib.util.spec_from_file_location(
                    schema_file.stem, schema_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find the main schema class (inherits from BaseModel)
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and issubclass(obj, BaseModel)
                        and obj != BaseModel
                    ):
                        self.schemas[name.lower()] = obj
                        logging.info(f"Loaded schema: {name}")
                        break

            except Exception as e:
                logging.error(f"Error loading schema {schema_file}: {e}")

    def get_schema_names(self) -> List[str]:
        """Get list of available schema names."""
        return list(self.schemas.keys())

    def get_schema(self, name: str) -> Type[BaseModel]:
        """Get schema class by name."""
        return self.schemas.get(name.lower())

    def save_new_schema(self, schema_name: str, schema_code: str) -> None:
        """Save a newly generated schema to file."""
        schema_path = Path(__file__).parent / f"{schema_name.lower()}.py"
        with open(schema_path, "w") as f:
            f.write(schema_code)
        logging.info(f"Saved new schema: {schema_path}")

        # Reload schemas to include the new one
        self._load_schemas()


# Create global schema manager instance
schema_manager = SchemaManager()

# Create ResponseOfSchema model dynamically with available schemas
schema_names = schema_manager.get_schema_names() + ["No match"]
ResponseOfSchema = create_model(
    "ResponseOfSchema", schema=(Literal[tuple(schema_names)], ...), reason=(str, ...)
)
