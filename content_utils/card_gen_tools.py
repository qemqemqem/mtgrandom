import json
import os
import random
import re

import requests

from content_utils.mechanics_balancer import generate_sets_with_target_complexity_str_to_strs
from content_utils.text_utils import remove_bullet_etc
from graphics_utils import dalle
from content_utils.llm import prompt_completion_chat
from set_logging.logger import log_generation_step
from llm_config import LLMModel

DETAILS_IN_ORDER = [
    'name',
    'manaCost',
    'rarity',
    'power',
    'toughness',
    'life',
    'loyalty',
    'defense',
    'attack',
    'type',
    # 'supertypes',
    # 'types',
    # 'subtypes',
    'text',
    'flavor',
    'artist']


def return_all_cards(atomic_cards_json):
    # Load the JSON file
    # Download this file from here: https://mtgjson.com/downloads/all-files/
    if not os.path.exists(atomic_cards_json):
        # Download the file if it doesn't exist
        url = 'https://mtgjson.com/api/v5/AtomicCards.json'
        response = requests.get(url)
        with open(atomic_cards_json, 'wb') as f:
            f.write(response.content)

    with open(atomic_cards_json, encoding='utf-8') as f:
        data = json.load(f)

    return list(data['data'].items())


def get_fake_example_card():
    # TODO I think the order of this is getting rearranged, which might matter for prompt engineering reasons
    return {
        "name": "Name of card",
        "supertype": "Sorcery, Land, Creature, Legendary Artifact, Enchantment, etc",
        "subtype": "Human, Elf, Warrior, etc",
        "power": "2",
        "toughness": "4",
        "rule_text": """Flying, Indestructible
{T}: Example tap ability
{1}{R}: Example mana ability""",
        "flavor_text": "A quote or vignette that adds flavor to the card",
        "mana_cost": "{2}{R}{W}",
        "rarity": "Common, Uncommon, or Rare",
    }


# Here's some advice about card ideas:
# - Commons should be simple and easy to understand
# - Uncommons should help support an archetype
# - Rares can be more complex and powerful
# - Many cards should connect to one of the mechanical archetypes
# - It's okay if some cards are good and flavorful on their own, as long as they can stand alone mechanically


def get_color_advice(card_idea, set_description):
    # This isn't as robust as I'd like it to be, and will fail if the guidance uses keywords like Azorius instead of Blue White
    mtg_colors = ["White", "Blue", "Black", "Red", "Green"]
    card_colors = []
    for color in mtg_colors:
        pattern = r'\b' + re.escape(color) + r'\b'
        if re.search(pattern, card_idea, re.IGNORECASE):
            card_colors.append(color)

    advice = ""
    if len(card_colors) == 2:
        # Find that exact color pair
        for line in set_description.split("\n"):
            perfect_match = True
            for color in card_colors:
                pattern = r'\b' + re.escape(color) + r'\b'
                if not re.search(pattern, line, re.IGNORECASE):
                    perfect_match = False
                    break
            if perfect_match:
                advice += remove_bullet_etc(line) + "\n"
    else:
        for line in set_description.split("\n"):
            for color in card_colors:
                pattern = r'\b' + re.escape(color) + r'\b'
                if re.search(pattern, line, re.IGNORECASE):
                    advice += remove_bullet_etc(line) + "\n"
                    break
    return advice


def generate_card(example, args, card_idea, mechanical_set_description):
    if example is None:
        example = get_fake_example_card()
        example_text = card_to_text(example, False)
    else:
        example_text = card_to_text(example[1][0], True)

    card_rarity = "Rare"
    if "uncommon" in card_idea.lower():
        card_rarity = "Uncommon"
    elif "common" in card_idea.lower():
        card_rarity = "Common"
    elif "rare" in card_idea.lower():
        card_rarity = "Rare"
    elif "mythic" in card_idea.lower():
        card_rarity = "Rare"

    rarity_guidance = """First, generate 5 simple mechanics, complexity 1-2, like "Hexproof" or "Lifelink".
Then, generate 5 medium complexity mechanics, complexity 2-4, like "When this creature enters the battlefield, draw a card"."""

    if card_rarity != "Common":
        rarity_guidance += "\nThen, generate 5 complex or weird mechanics that might take 1 or 2 lines of rules text, complexity 4-5."

    advice = get_color_advice(card_idea, mechanical_set_description)

    prompt = f"""Please generate a card. Here is the idea I have for it: 

{card_idea}

# Set Mechanics

{advice}

Based on this set description, what would be synergistic abilities for this card to have?

# Brainstorming

I want you to brainstorm 15 possible mechanics for this card. 

For each possible mechanic, write a short description of how it would work, as though you were writing Oracle text for the card, like "when this card enters the battlefield, its controller draws a card". Mention a card that has this mechanic, if you can think of one. 

Then, rate the complexity of the mechanic on a scale from 1-5. 1 is simple, like a common keyword, like "Flying" or "Haste". 3 is an unusual ability, like unblockable or hexproof. 5 is a very complex ability that requires a lot of rules text.

Then, rate how well the mechanic supports the flavor of the card on a scale from 1-5. 1 means the mechanic doesn't support the flavor at all. 5 means the mechanic is a perfect fit for the flavor.

Then, rate how well the mechanic supports the mechanical archetype of the card on a scale from 1-5. 1 means the mechanic doesn't support the archetype at all. 5 means the mechanic is a perfect fit for the archetype.

{rarity_guidance}

Again, the idea for the card is: {card_idea}

Put each possible mechanic on its own line, like this:

1. Text of the mechanic. Similar to [Card]. Complexity X. Flavor X. Synergy X."""
    messages = [{"role": "system", "content": f"You are a game designer creating Magic the Gathering cards. You love good mechanics and good gameplay."},
                {"role": "user", "content": prompt},]

    suggested_mechanics_str = prompt_completion_chat(messages=messages, n=1, temperature=0.0, max_tokens=1512, model_id=args.llm_model)

    guess_card_name = remove_bullet_etc(card_idea)[:remove_bullet_etc(card_idea).index(".")]
    if ":" in guess_card_name:
        guess_card_name = guess_card_name[:guess_card_name.index(":")]
    log_generation_step("suggested mechanics", prompt, suggested_mechanics_str, args.set_name if args else None, guess_card_name)

    suggested_mechanics = []
    for line in suggested_mechanics_str.split("\n"):
        if line.strip() == "":
            continue
        if line[0] == "#":
            continue
        if "complexity" not in line.lower() and "flavor" not in line.lower() and "synergy" not in line.lower():
            continue
        if "simple mechanics" in line.lower() or "medium complexity mechanics" in line.lower() or "complex or weird mechanics" in line.lower() or "complex mechanics" in line.lower():
            continue
        suggested_mechanics.append(remove_bullet_etc(line))
    suggested_mechanics_str = "\n".join(suggested_mechanics)

    target_complexity = 7  # For rares
    max_num_mechanics = 3
    if card_rarity == "Common":
        target_complexity = 3
        max_num_mechanics = 2
    elif card_rarity == "Uncommon":
        target_complexity = 5
    suggested_mechanics_sets = generate_sets_with_target_complexity_str_to_strs(suggested_mechanics, target_complexity, 3)

    mechanics_sets_str = ""
    for i, mechanics_set in enumerate(suggested_mechanics_sets):
        mechanics_sets_str += f"Design Idea{i+1}.\n{mechanics_set}\n\n"

    messages = [{"role": "system", "content": f"You are a game designer creating Magic the Gathering cards. You love good mechanics and good gameplay."},
                {"role": "user", "content": f"""I need help generating a Magic the Gathering card. Here is the idea I have for it: 

{card_idea}

Here's what its colors are like in the set:

{advice}

# Mechanics

I have some ideas for mechanics, but I'm not sure which ones to use. Here are some mechanics I'm considering:

{suggested_mechanics_str}

# Possible Card Ideas

I want the card to use a few of these mechanics, in a nice combination that's flavorful and synergistic. I was thinking something like this:

{mechanics_sets_str}

I'm worried that these mechanics might be too complicated.

# Final Card Design

Can you think about those mechanics and the designs I've thought of, and come up with the final best set of mechanics for this card? Explain why the design you've chosen is synergistic.

The card should have no more than {max_num_mechanics} mechanics, and should have a total complexity of no more than {target_complexity}.

Again, the card idea is: {card_idea}

Please think about the design and then say "# Final Card" and then give a card design."""},]

    final_card_mechanics_idea = prompt_completion_chat(messages=messages, n=1, temperature=0.2, max_tokens=1512, model_id=args.llm_model)
    print(final_card_mechanics_idea)

    log_generation_step("final card mechanics idea", messages[1]["content"], final_card_mechanics_idea, args.set_name if args else None, guess_card_name)

    messages = [{"role": "system", "content": f"You are a game designer creating Magic the Gathering cards. You love good mechanics and good gameplay."},
                {"role": "user", "content": f"Please show me the format for a Magic the Gathering card."},
                {"role": "assistant", "content": f"```json\n{example_text}\n```"},
                {"role": "user", "content": f"""Please generate a card. Here is the idea I have for it: 

{card_idea}

And here is the mechanical design I want for the card:

{final_card_mechanics_idea}

First, decide on the mana cost for the card, unless it's a land. 

Then, decide on the card's power and toughness, if it's a creature.

Make sure you know it's supertype and subtype, as well as its rarity and name. 

If it's a planeswalker or other card type, decide on any other attributes it needs.

Make sure the card mechanics are in oracle text.

Finally, in the same JSON format that you showed me above, write out the full details of the card."""},]

    final_card = prompt_completion_chat(messages=messages, n=1, temperature=0.2, max_tokens=1512, model_id=args.llm_model)
    print(final_card)

    log_generation_step("final card json", messages[3]["content"], final_card, args.set_name if args else None, guess_card_name)

    return final_card, suggested_mechanics_str


def criticize_and_try_to_improve_card(card, args, suggested_mechanics_str):
    # TODO(andrew) I think I need to totally redo this prompt.
    # It should first focus on missing content and rules issues, which are the most serious problems.
    # I should include a list of the most common design issues, like parasitic mechanics, and ask it to identify those
    # I should include a scoring system like "Rate complexity on a scale of 1-5" and "Rate power level on a scale of 1-5"
    # I should give it clear rules for when to accept or reject a card based on those scores, like commons should be a 1-3 on both, uncommons should be a 2-4, and rares should be a 3-5
    # It needs special guidance for lands, such as the need to give them a downside if they tap for 2 or more colors

    specific_advice = ""

    if "land" in card["supertype"]:
        specific_advice += "* Lands need to have a downside, like entering the battlefield tapped, requiring the player to pay {1} when it enters, or costing life to use. Being legendary can be a small downside, but if it taps for any color of mana, it needs a big downside. If this card is a land, write \"Downside: [cost to use here]\". If this land card doesn't have a downside, write \"Needs work: land needs downside\" and suggest a potential cost\n\n"

    messages = [{"role": "system", "content": f"You generate Magic the Gathering cards. You are not afraid to be critical."},
                {"role": "user", "content": f"""I want help designing a Magic the Gathering card. Here are the details:

```json
{card_to_text(card)}
```

Please answer these questions about this card, and give constructive criticism:

* Name one or two cards that are mechanically similar to this card, for reference.

* Is it missing any important details like mana cost, or power and toughness if it's a creature? If so, write "Needs work: Missing details" and write what's missing

* Does the text on the card make sense mechanically? Is it possible to do what the card says? If not, write "Needs work: Mechanical Issues"

* Is it written in the right style for MTG? Would a judge have trouble interpreting it? If it's not written in the style of Oracle text, write "Needs work: Wrong Style"

{specific_advice}Rate the power level of the card on a scale from 0-6, where 0 is unplayable, 1 is very weak, like a bad common, 5 is very strong, like a bomb rare, and 6 is super overpowered like a black lotus. We're aiming for these power levels:
Commons: 1-3
Uncommons: 2-4
Rares: 3-5

Rate the complexity of the card on a scale from 0-6, where 0 is a vanilla creature, 1 is very simple, 5 is very complex, like a planeswalker, and 6 is way too confusingly complicated. We're aiming for these complexity levels:
Commons: 1-3
Uncommons: 2-4
Rares: 3-5

If we're not within those ranges, write "Needs work: Power Level too [high, low]" or "Needs work: [Too complex, too simple]"

Rate the flavor of the card and the match between the flavor and the mechanics on a scale from 1-5, where 1 is a boring card or a card whose theme is not at all reflected in the mechanics, and 5 is a card with a very interesting theme that is well reflected in the mechanics. If the card is a 1 or a 2, write "Needs work: Flavor".

If the card passes all these tests, then great! Please write "Looks good".

For now, just answer the questions."""},]
    criticism = prompt_completion_chat(messages=messages, n=1, temperature=0.0, max_tokens=512, model_id=args.llm_model)

    log_generation_step("please criticize card", messages[1]["content"], criticism, args.set_name if args else None, card["name"] if "name" in card else "Unknown Card")

    looks_good: bool = False
    if "looks good" in criticism.lower():
        looks_good = True

    if "needs work" in criticism.lower():
        looks_good = False
        for line in criticism.split("\n"):
            if "needs work" in line.lower():
                print("Fixing:", line)
                break

    if not looks_good:
        messages = [{"role": "system",
                     "content": f"You generate Magic the Gathering cards. You are not afraid to be critical."},
                    {"role": "user", "content": f"""I want help designing a Magic the Gathering card. Here are the details I have for the card:

```json
{card_to_text(card)}
```

What do you think of this card?"""},
                    {"role": "assistant", "content": f"{criticism}"},
                    {"role": "user", "content": f"""Given your feedback, please try to improve the card. Please output JSON for the improved card.

Here are some other mechanics we could consider for the card, if we really need to change it based on that criticism:

{suggested_mechanics_str}

Here are the details of the card that needs to be fixed, again:

```json
{card_to_text(card)}
```"""}, ]
        improved_card = prompt_completion_chat(messages=messages, n=1, temperature=0.0, max_tokens=512, model_id=args.llm_model)

        log_generation_step("improved card", messages[3]["content"], improved_card, args.set_name if args else None, card["name"] if "name" in card else "Unknown Card")

        improved_card_dict = generate_dict_given_text(improved_card)
        return improved_card_dict, False
    else:
        return card, True


def load_card_ideas(card_names_file):
    with open(card_names_file, encoding='utf-8') as f:
        card_names = f.readlines()
    return [x.strip() for x in card_names if x.strip() != ""]


def card_to_text(card, fix_inclusions=False):
    if fix_inclusions:
        if 'rarity' not in card:
            card['rarity'] = 'Uncommon'
        simpler_card = {}
        for detail in DETAILS_IN_ORDER:
            if detail in card:
                simpler_card[detail] = card[detail]
    else:
        simpler_card = card
    s = json.dumps(simpler_card, indent=4)
    return s


def generate_dict_given_text(text):
    # TODO This needs to be improved. For example, sometimes it doesn't put everything on one line

    text = text.strip()
    if text == "":
        return {}
    details = {}
    try:
        if "```json" in text:
            potential_json = text[text.find("```json")+7:text.rfind("```")]
        else:
            potential_json = text[text.find('{'):text.rfind('}')+1]
        details = json.loads(potential_json)
        print(details)
    except:
        print("SAD! Couldn't parse as JSON: ", text)
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if line == "":
                continue
            if line.count(":") == 0:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key in ["supertypes", "types", "subtypes"]:
                details[key] = value.split(",")
            else:
                details[key] = value

    # Try to fix it up
    # Lower case everything
    keys_copy = list(details.keys())
    for key in keys_copy:
        if key != key.lower():
            details[key.lower()] = details[key]
            del details[key]
    # Fix multi-word keys
    for flavor_synonym in ["flavor text", "flavortext"]:
        if flavor_synonym in details:
            details["flavor"] = details[flavor_synonym]
            del details[flavor_synonym]
    for artist_synonym in ["artist name", "artistname"]:
        if artist_synonym in details:
            details["artist"] = details[artist_synonym]
            del details[artist_synonym]
    for mana_synonym in ["mana cost", "manacost"]:
        if mana_synonym in details:
            details["manaCost"] = details[mana_synonym]
            del details[mana_synonym]

    # For some reason it sometimes has "abilities" instead of "rule_text"
    if "abilities" in details:
        print("Got abilities, trying to repair:", details)
        if "rule_text" in details:
            print("Weird, found both abilities and rule_text")
        else:
            ability_texts = []
            for ability in details["abilities"]:
                if ability is str:
                    details["rule_text"] += ability
                elif ability is dict:
                    main_ability = ability["text"] if "text" in ability else ability["effect"] if "effect" in ability else ability["description"] if "description" in ability else ""
                    if "cost" in ability:
                        if ability["cost"] in main_ability:
                            # The text is duplicated
                            ability_texts.append(main_ability)
                        else:
                            ability_texts.append(ability["cost"] + ": " + main_ability)
                    else:
                        ability_texts.append(main_ability)
            details["rule_text"] = "\n".join(ability_texts)

    # Should have power and toughness
    is_creature = False
    if "type" in details:
        is_creature = "creature" in details["type"]
    elif "supertype" in details:
        is_creature = "creature" in details["supertype"]
    if is_creature and ("power" not in details or "toughness" not in details):
        raise Exception("Creature card without power and toughness")

    return details



if __name__ == '__main__':
    pass