
# MTG Random Card Generator

Generate custom Magic: The Gathering cards. Use ChatGPT or another LLM to generate JSON card details, using existing cards as examples (and to cue the formatting). Then, use an image generation AI such as DALLE to generate art for the cards. Full cards are then rendered using the generated JSON and images.

⭐ If you find this project helpful, please consider giving it a star! ⭐

See `sets/example/cards` for examples of generated cards.

I hope you enjoy!

This project is very incomplete, so I welcome any contributions.

## Project Overview

The project is structured into these main parts:

1. **Set Generation (`set`)**: Attempt to generate a cohesive set, with themes and mechanics. 

1. **Card JSON Generation (`cards`)**: This part of the project generates the card details in a JSON format. It utilizes a pre-existing dataset, optionally allows for custom card names, and makes use of AI to generate card details.
   
1. **Image Generation (`images`)**: This step uses a generative graphics model to produce captivating images for the generated cards.

1. **Full Card Generation (`full`)**: This combines the card details with the generated images to render full Magic card designs.

## How to Run

1. **Prerequisites**:
   - Ensure Python is installed on your system.
   - Install required Python libraries by running:
     ```
     pip install -r requirements.txt
     ```
    - Download the [AtomicCards.json](https://mtgjson.com/downloads/all-files/) file from MTGJSON and place it in the root directory of the project.
    - Get an API key from Open AI and `export OPENAI_API_KEY=your_key_here`.

2. **Command Line Arguments**:
   The script accepts a range of command line arguments to customize the card generation process:
   - `action`: Choose which part of the process to run (choices: `set`, `cards`, `images`, `full`, `all`).
   - `--set-name`: Name of the set. Required.
   - `--set-description`: Description of the set. Required if generating a new set.
   - `--atomic-cards-file`: Path to AtomicCards.json.
   - `--set-size`: If all or set, the size of the set to generate. 
   - `--max-cards-generate`: Maximum number of cards to generate. This may be smaller than the set size. If 0, no max.
   - `--llm-model`: LLM model to use. Choices: gpt-3.5-turbo, gpt-4
   - `--graphics-model`: Graphics model to use. Currently supports either `dalle` or `midjourney`

3. **Execution**:

   - To generate a set:
     ```
        python main.py set --set-name "new_set" --set-description "This is a new set."
     ```
     or
     ```
        python main.py set --set-name "dune" --set-description "An MTG set inspired by Frank Herbert's Dune." --set-size 36
     ```
   
   - To generate card details in JSON format:
     ```
     python main.py cards --set-name "new_set" --max-cards-generate 5
     ```
     
   - To generate card images (for all cards in the set):
     ```
     python main.py images --set-name "new_set" --graphics-model "dalle"
     ```
     
   - To render the complete cards with images (for all cards in the set):
     ```
     python main.py full --set-name "new_set"
     ```
   
   - To run all the above steps in sequence:
     ```
     python main.py all --set-name "new_set" --set-description "A cool game about wizards and caves and monsters and stuff." --set-size 12 --max-cards-generate 12 --llm-model gpt-3.5-turbo --graphics-model dalle
     ```

---

# TODO

This project isn't done. Here are the major things that could be improved:

Set Design:

- [ ] It would be nice to generate mechanics for the set, and generally do some set design, before going in and generating cards. This is partly done but could be better
- [ ] Review the set for synergies and combos.

Card Generation:

- [ ] Go through AtomicCards.json to find similar cards to the generated one, for comparison. For example, find 3 cards with the same color, manacost, type, and rarity, and compare the generated card to those.
- [ ] Improve LLM criticism of the cards it's generating, and do more revisions and edits
- [ ] It would be nice if there was a "set-size" parameter, which would generate cards up to that number, rather than always generating that many new cards, like --number-of-cards-to-generate does.

Art Generation:

- [ ] Connect another image generator, like Midjourney. We can use [this one](https://github.com/yachty66/unofficial_midjourney_python_api). 
- [ ] Have the LLM generate the art descriptions, including suggesting an artist. We can credit the artist on the final card with "insp. by [artist]".

Full Card Rendering:
- [ ] Improve the final card rendering. Right now they look quite ugly.
- [ ] In particular, mana symbols aren't rendering properly.

