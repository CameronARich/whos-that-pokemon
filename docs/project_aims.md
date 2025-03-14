# Who's that Pokémon?

## Given:

/* _Obtained per [Noodulz on Kaggle](https://www.kaggle.com/datasets/noodulz/pokemon-dataset-1000/data)_ */
The 1000 Pokemon Dataset:

> This dataset contains ~40 images per 1,000 Pokémon species, structured in subdirectories for each class. Each image is resized to 128x128 pixels and stored as a PNG file. Originally made for a Pokedex project in Flutter. Data augmentation highly recommended when using this for training a model. 

> Total images: 26,539

> Total classes: 1,000

> Size: ~407MB

### (OPTIONAL)

/* _Obtained per [Veekun](https://veekun.com/dex/downloads)_ */
Official art for generations 1-5 of Pokémon (#1-#649): 
- all battle sprites for each Pokémon in each game (`./data/sprites`)
- their official Ken Sugimori renders (`./data/renders`)

***

## Research and test:

1. How well sklearn computer vision methods can match Pokémon

  - (optional) and possibly, each generation of pixel spritework, as well as their official renders.

2. How each of the methods used compare with each other, accuracy-wise.

3. How generational differences affect these methods.
    - Are Pokémon more inclined to map to their original sprites or their updated sprites?
    - Does the 2-bit (hue-corrected?) color of the Gen 1/2 sprites kill color matching?
    - How does increasing color and sprite fidelity in each generation correlate with ability to match?