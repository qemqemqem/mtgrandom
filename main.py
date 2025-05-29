import os
import argparse
import json
import random

from content_utils.art_director import get_art_prompt
from content_utils.flavor_writer import write_flavor_for_card
from content_utils.set_gen import generate_set_description, generate_card_suggestions
from graphics_utils import render_full_card, midjourney
from content_utils.card_gen_tools import *
from mse import mse_gen
from llm_config import LLMModel


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate Cards")

    parser.add_argument('action', choices=['set', 'cards', 'images', 'full', 'all'], help='Action to be performed.')
    parser.add_argument('--set-name', default='testing', help='Name of the set.')
    parser.add_argument('--set-description', default='Cool fantasy world but with funny animals', help='Description of the set.')
    parser.add_argument('--card-names-file', default=None, help='File with card names.')
    parser.add_argument('--atomic-cards-file', default='AtomicCards.json', help='Path to AtomicCards.json.')
    parser.add_argument('--max-cards-generate', default=0, type=int, help='Maximum number of cards to generate. If zero, no max.')
    parser.add_argument('--set-size', default=18, type=int, help='The size of the set.')
    parser.add_argument('--llm-model', default='BEST_WRITING_MODEL', help='LLM model to use.')
    parser.add_argument('--graphics-model', default='dalle', help='Graphics model to use. Options: dalle, midjourney')
    parser.add_argument('--art-guidance', default='', help='Art guidance for the graphics model.')
    parser.add_argument('--art-only', action='store_true', help='Only generate art.')
    parser.add_argument('--mse-location', default='', help='Location of your mse.exe file, the Magic Set Editor. Used by the "full" action. If empty, cards will be rendered with an ugly HTML based method. If you\'re on Linux, use "wine mse.exe".')

    args = parser.parse_args()
    args.llm_model = LLMModel[args.llm_model]
    return args


def generate_set(args):
    set_description_file_name = f"sets/{args.set_name}/set_description.txt"
    if os.path.exists(set_description_file_name):
        print(f"Set description file {set_description_file_name} already exists. Skipping.")
    else:
        set_description = generate_set_description(args)
        args.set_description = set_description
        with open(set_description_file_name, "w", encoding="utf-8") as f:
            f.write(set_description)

    # Generate some card ideas
    card_suggestions_file_name = f"sets/{args.set_name}/card_suggestions.txt"
    num_cards_to_generate_suggestions_for = args.set_size
    if os.path.exists(card_suggestions_file_name):
        with open(card_suggestions_file_name, "r", encoding="utf-8") as f:
            card_suggestions = f.readlines()
            number_of_existing_suggestions = len([suggestion for suggestion in card_suggestions if suggestion.strip() != ""])
            num_cards_to_generate_suggestions_for = max(0, num_cards_to_generate_suggestions_for - number_of_existing_suggestions)
        print(f"Card suggestions file {card_suggestions_file_name} already exists. Generating {num_cards_to_generate_suggestions_for} more suggestions.")
    if num_cards_to_generate_suggestions_for > 0:
        card_suggestions, story = generate_card_suggestions(args, num_cards_to_generate_suggestions_for)
        with open(card_suggestions_file_name, "a", encoding="utf-8") as f:
            for card_suggestion in card_suggestions:
                f.write(card_suggestion + "\n")
        with open(f"sets/{args.set_name}/story.txt", "w", encoding="utf-8") as f:
            f.write(story)


def generated_cards_json(args):
    # all_cards = return_all_cards(args.atomic_cards_file)
    num_generated_cards = 0
    with open(f"sets/{args.set_name}/cards.jsonl", "a") as f:
        pass  # Just touch the file
    card_suggestions_file = f"sets/{args.set_name}/card_suggestions.txt"
    if os.path.exists(card_suggestions_file):
        new_card_ideas = load_card_ideas(card_suggestions_file)
    else:
        adjectives = ["Cute", "Funny", "Silly", "Goofy", "Weird", "Strange", "Bizarre", "Unusual", "Quirky", "Odd", "Angry", "Sad", "Happy", "Frenzied", "Fantastic", "Questing", "Lost", "Forgotten", "Ancient", "Charming", "Enchanted", "Mysterious", "Running"]
        creatures = ["Badger", "Peacock", "Wizard", "Barbarian", "Emu", "Penguin", "Panda", "Wallaby", "Koala", "Kangaroo", "Dingo", "Dinosaur", "Dragon", "Unicorn", "Pegasus", "Griffin", "Phoenix", "Gryphon", "Goblin", "Orc", "Troll", "Ogre", "Elf", "Fairy", "Mermaid", "Centaur", "Minotaur", "Satyr", "Giant", "Gnome", "Golem", "Gargoyle", "Demon", "Angel", "Vampire", "Werewolf", "Zombie", "Skeleton", "Ghost", "Specter", "Goose"]
        new_card_ideas = [f"{random.choice(adjectives)} {random.choice(creatures)}" for _ in range(args.number_of_cards_to_generate)]
        # random.shuffle(new_card_ideas)
    mechanical_set_description_file = f"sets/{args.set_name}/set_description.txt"
    if os.path.exists(mechanical_set_description_file):
        with open(mechanical_set_description_file, "r") as f:
            mechanical_set_description = f.read()
    else:
        mechanical_set_description = "Cool fantasy world but with funny animals"
    story_file = f"sets/{args.set_name}/story.txt"
    if os.path.exists(story_file):
        with open(story_file, "r") as f:
            story = f.read()
    else:
        story = "Story unknown"
    for i, card_idea in enumerate(new_card_ideas):
        card_idea = remove_bullet_etc(card_idea)
        if 0 <= args.max_cards_generate <= num_generated_cards:
            break
        approx_card_name = remove_bullet_etc(card_idea[:card_idea.find(".")])
        if "(" in approx_card_name:
            approx_card_name = approx_card_name[:approx_card_name.find("(")].strip()
        approx_card_name = re.sub(r"[!@#$%^&*]", "", approx_card_name)
        with open(f"sets/{args.set_name}/cards.jsonl", "r") as f:
            full_cards_jsonl = f.read()
            if f"name\": \"{approx_card_name}" in full_cards_jsonl:
                print(f"Skipping card {i + 1} (Already exists) out of {len(new_card_ideas)}: {card_idea}")
                continue
        # example_card = random.choice(all_cards)
        # I found that providing an example card did not help with diversity prompting and sometimes led to poor formatting

        print(f"Generating card {i + 1} out of {len(new_card_ideas)}: {card_idea}")
        generated, suggested_mechanics_str = generate_card(None, args, card_idea, mechanical_set_description)
        print("-" * 80)
        print(f"Generated card {i + 1} out of {len(new_card_ideas)}: {card_idea}: {generated}")
        num_generated_cards += 1
        print(generated)
        generated_dict = generate_dict_given_text(generated)

        # Iteratively criticize and improve the card
        num_fix_iterations = 4
        if args.art_only:
            num_fix_iterations = 0
        for fix_iteration in range(num_fix_iterations):
            # TODO Consider presenting all iterations here
            print(f"Fixing card {i + 1} out of {len(new_card_ideas)}, Iteration: {fix_iteration + 1}/{num_fix_iterations}")
            generated_dict, looks_good = criticize_and_try_to_improve_card(generated_dict, args, suggested_mechanics_str)
            print("Fixed and improved card:", generated_dict)
            if looks_good:
                print("Looks good!")
                break
        generated_dict['art_prompt'], generated_dict['artist_credit'] = get_art_prompt(generated_dict, args)
        generated_dict['flavor_text'] = write_flavor_for_card(card_idea, generated_dict, story, args)
        with open(f"sets/{args.set_name}/cards.jsonl", "a") as f:
            f.write(json.dumps(generated_dict) + "\n")


def generated_cards_images(args):
    os.makedirs(f"sets/{args.set_name}/images", exist_ok=True)
    generated_images = 0
    with open(f"sets/{args.set_name}/cards.jsonl", "r") as f:
        cards = [json.loads(line) for line in f.readlines()]
        for i, card in enumerate(cards):
            if 0 <= args.max_cards_generate <= generated_images:
                break
            # flavor = card['flavor'] if 'flavor' in card else (card['text'] if 'text' in card else "")
            image_path = f"sets/{args.set_name}/images/{card['name']}.png"
            # art_prompt = f"{card['name']}, Magic the Gathering Art, Beautiful, Fantasy, Spec Art. {flavor}"
            if not os.path.exists(image_path):
                print(f"Generating image for card {i+1} out of {len(cards)}:", card['name'])
                generated_images += 1
                if 'art_prompt' in card:
                    art_prompt = card['art_prompt']
                else:
                    art_prompt, _ = get_art_prompt(card, args)
                if args.graphics_model == "dalle":
                    dalle.generate_image_and_save_to_file(art_prompt, image_path)
                elif args.graphics_model == "midjourney":
                    midjourney.generate_image_and_save_to_file(art_prompt, image_path)
                else:
                    print("Unknown graphics model.", args.graphics_model)
            else:
                print(f"Image already exists for card {i+1} of {len(cards)}:", card['name'])


def generate_full_card_images(args):
    if args.mse_location is None or args.mse_location == "":
        os.makedirs(f"sets/{args.set_name}/cards", exist_ok=True)
        with open(f"sets/{args.set_name}/cards.jsonl", "r") as f:
            cards = [json.loads(line) for line in f.readlines()]
            for card in cards:
                image_path = f"sets/{args.set_name}/images/{card['name']}.png"
                card['image_path'] = image_path
                if not os.path.exists(f"sets/{args.set_name}/cards/{card['name']}.png"):
                    print(card)
                    print("-" * 80)
                    render_full_card.create_magic_card(card, f"sets/{args.set_name}")
    else:
        mse_gen.load_and_create_set(args.set_name, args.mse_location)


def main():
    args = parse_arguments()
    os.makedirs(f"sets/{args.set_name}", exist_ok=True)

    set_description_file_name = f"sets/{args.set_name}/set_description.txt"
    if os.path.exists(set_description_file_name):
        args.full_set_guidelines = open(set_description_file_name, "r", encoding="utf-8").read()

    if args.action == "set" or args.action == "all":
        print("Generating set...")
        generate_set(args)
    if args.action == "cards" or args.action == "all":
        print("Generating cards...")
        generated_cards_json(args)
    if args.action == "images" or args.action == "all":
        print("Generating images...")
        generated_cards_images(args)
    if args.art_only:
        print("Art only mode, skipping full card images.")
        return
    if args.action == "full" or args.action == "all":
        print("Generating full card images...")
        generate_full_card_images(args)


if __name__ == '__main__':
    main()
