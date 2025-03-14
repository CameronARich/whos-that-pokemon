#!/usr/bin/env python3
import os
import re
import requests
import time
from pathlib import Path
import argparse

# Cache for Pokémon data to avoid hitting the API repeatedly
pokemon_cache = {}

def get_pokemon_name(pokemon_id):
    """
    Get Pokémon name from pokeapi.co based on the ID
    Uses caching to avoid repeated API calls
    """
    if pokemon_id in pokemon_cache:
        return pokemon_cache[pokemon_id]
    
    # Rate limiting to be respectful of the API
    time.sleep(0.5)
    
    try:
        url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_id}"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            # Get the Pokémon name, casefold it, and convert spaces to underscores
            name = data['species']['name'].lower().replace(' ', '-').replace('_', '-')
            
            # Store in cache for future use
            pokemon_cache[pokemon_id] = name
            return name
        else:
            print(f"Failed to get data for Pokémon ID {pokemon_id}. Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error fetching Pokémon data for ID {pokemon_id}: {e}")
        return None

def format_filename(pokemon_name, suffix):
    """
    Format the new filename correctly based on Pokémon name and suffix
    """
    # If the suffix starts with a dash and then text (like "-east"),
    # we'll want to keep the dash but make the text lowercase and snake_case
    if suffix and suffix.startswith('-'):
        # Handle special forms while keeping the dash
        form = suffix[1:]  # Remove the dash
        
        # Split by additional dashes if present
        form_parts = form.split('.')
        if len(form_parts) > 1:
            # Handle file extension
            extension = form_parts[-1]
            form_name = '.'.join(form_parts[:-1])
            
            # Make form name lowercase and convert spaces to underscores
            if form_name:
                form_name = form_name.lower().replace(' ', '_')
                return f"{pokemon_name}-{form_name}.{extension}"
            else:
                return f"{pokemon_name}.{extension}"
        else:
            # No file extension in the suffix (unusual)
            return f"{pokemon_name}{suffix}"
    else:
        # Just a simple suffix or extension
        return f"{pokemon_name}{suffix}"

def rename_pokemon_file(filepath):
    """
    Rename a file by replacing the leading number with the Pokémon name
    Returns True if successful, False otherwise
    """
    filename = os.path.basename(filepath)
    directory = os.path.dirname(filepath)
    
    # Regex to match patterns like:
    # - Just a number (e.g., "25.png")
    # - Number followed by a dash and text (e.g., "423-east.png")
    match = re.match(r'^(\d+)(.*)$', filename)
    
    if match:
        pokemon_id = match.group(1)
        suffix = match.group(2)
        
        # Get Pokémon name from API
        pokemon_name = get_pokemon_name(pokemon_id)
        
        if pokemon_name:
            # Create new filename with proper formatting
            new_filename = format_filename(pokemon_name, suffix)
            new_filepath = os.path.join(directory, new_filename)
            
            # Check if new file already exists to avoid overwriting
            if os.path.exists(new_filepath):
                print(f"Cannot rename {filepath} to {new_filepath} - file already exists")
                return False
            
            # Rename the file
            try:
                os.rename(filepath, new_filepath)
                print(f"Renamed: {filepath} -> {new_filepath}")
                return True
            except Exception as e:
                print(f"Error renaming {filepath}: {e}")
                return False
        else:
            print(f"Could not get Pokémon name for ID {pokemon_id} in file {filepath}")
            return False
    else:
        # File doesn't start with a number, skip it
        return False

def process_directory(directory):
    """
    Recursively process all files in a directory
    """
    renamed_count = 0
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            # Skip macOS system files
            if filename.startswith('.'):
                continue
                
            filepath = os.path.join(root, filename)
            if rename_pokemon_file(filepath):
                renamed_count += 1
    
    return renamed_count

def main():
    """Main function to parse arguments and process directories"""
    parser = argparse.ArgumentParser(description='Rename Pokémon files replacing IDs with names')
    parser.add_argument('directories', nargs='+', help='Directories to process')
    parser.add_argument('--dry-run', action='store_true', help="Preview changes without actually renaming files")
    
    args = parser.parse_args()
    
    # If dry run is enabled, we'll modify the rename_pokemon_file function to just print
    if args.dry_run:
        global rename_pokemon_file
        original_rename = rename_pokemon_file
        
        def dry_run_rename(filepath):
            filename = os.path.basename(filepath)
            directory = os.path.dirname(filepath)
            
            match = re.match(r'^(\d+)(.*)$', filename)
            if match:
                pokemon_id = match.group(1)
                suffix = match.group(2)
                pokemon_name = get_pokemon_name(pokemon_id)
                
                if pokemon_name:
                    new_filename = format_filename(pokemon_name, suffix)
                    new_filepath = os.path.join(directory, new_filename)
                    print(f"Would rename: {filepath} -> {new_filepath}")
                    return True
                return False
            return False
        
        rename_pokemon_file = dry_run_rename
    
    total_renamed = 0
    for directory in args.directories:
        if not os.path.isdir(directory):
            print(f"Warning: {directory} is not a valid directory, skipping")
            continue
            
        print(f"Processing directory: {directory}")
        renamed = process_directory(directory)
        total_renamed += renamed
        print(f"Processed {renamed} files in {directory}")
    
    action = "Would rename" if args.dry_run else "Renamed"
    print(f"{action} a total of {total_renamed} files")

if __name__ == "__main__":
    main() 