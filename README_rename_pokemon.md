# Pokémon File Renamer

This script renames files in a directory by replacing the leading Pokémon ID numbers with their corresponding names using the [PokéAPI](https://pokeapi.co/).

## Features

- Recursively traverses directories to find and rename all Pokémon image files
- Handles special cases like files with form names (e.g., "423-east.png")
- Uses caching to minimize API calls
- Includes a dry-run option to preview changes without applying them
- Respects the PokéAPI rate limits to avoid being blocked

## Requirements

The script requires the `requests` library. If you don't have it installed, you can install it with:

```bash
pip install requests
```

## Usage

```bash
# Run with dry-run to see what would happen without changing files
python rename_pokemon_files.py --dry-run data/renders data/sprites

# Run the actual rename operation
python rename_pokemon_files.py data/renders data/sprites

# You can also specify a single directory
python rename_pokemon_files.py data/renders
```

## Example

A file named `25.png` will be renamed to `pikachu.png`
A file named `423-east.png` will be renamed to `gastrodon-east.png`

## Notes

- The script skips files that don't start with a number
- The script includes rate limiting to be respectful of the PokéAPI service
- If a file with the new name already exists, the script will not overwrite it
- Pokémon names are formatted in lowercase with spaces converted to underscores (snake_case)
- Form names (like "-east") are kept with the hyphen but also converted to lowercase 