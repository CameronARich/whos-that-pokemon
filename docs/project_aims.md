# Who's that Pokémon?

## Given:

/* _Obtained per [Veekun](https://veekun.com/dex/downloads)_ */
Within generations 1-5 of Pokémon (#1-#649)--a folder ./data containing: 
- all battle sprites for each Pokémon in each game. 
- their official Ken Sugimori renders.

and possibly: 
- the "Who's that Pokémon" eyecatches from the Pokémon anime. /* _(explore later maybe? not as of now)_ **--Cam** */ 

***

## Research and test:

1. How well computer vision methods can correlate the pixel spritework with official drawn renders.

    **Potential methods:**
        1. Color comparison /* _will likely not work well with Gen 1/2 sprites, but good to test_ **--Cam** */
        2. Masks, shapes, contours? /* _unsure about exact implementation but the most gen-agnostic imo_ **-- Cam** */

2. How each of the methods used compare with each other, accuracy-wise.

