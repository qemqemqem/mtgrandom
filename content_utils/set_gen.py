import os
import re

from content_utils.llm import prompt_completion_chat
from content_utils.set_balancer import create_balanced_set
from content_utils.text_utils import remove_bullet_etc
from set_logging.logger import log_generation_step
from content_utils.llm_config import LLMModel


def generate_set_description(args):
    prompt = f"""I'm an employee at Wizards of the Coast and I'm designing a new set of Magic: The Gathering cards.

The theme of the set is: {args.set_description}

To start with, let's describe some possible mechanics that this set might employ. First, brainstorm a brief list of 10 existing mechanics that might be appropriate for the setting. In one sentence or less, explain what the mechanic means.

Then, for each of those possible mechanics, describe a previous Magic the Gathering set that has used that mechanic. Describe how they used that mechanic, and what made it fun to draft and play.

Then describe some of the common mechanics that show up in Magic core sets, like flying or creature type synergies, and how they might be used in this set. You don't need to discuss all of them, but mention anything that you think is particularly relevant to the theme.

Then briefly speculate on 10 synergies or deck archetypes that the set might feature. Focus your description on the gameplay aspect here. For each synergy or archetype, describe how it played in a previous set. I want this to be a strong draft set, so focus your analysis on how these synergies and archetypes will play in draft.

Restrictions: No double sided cards such as werewolfs, no cards that transform, no cards that have a different back.

Next, please discuss how different themes and mechanics will be distributed on the Magic color pie, because we want the set to be balanced across colors. Write one line for each of the five colors describing their theme and mechanics.

Then, write one sentence giving guidance for each of the ten two-color pairs. I want each pair of colors to have a direction during the draft. That might be a mechanic like graveyard synergies, +1/+1 counters, "spells matter" or vehicles. These directions will help us design the cards.

If this set idea reminds you of any existing cards, please mention them, but be brief."""
    print(f"Generating set description for {args.set_name}")
    messages = [{"role": "system", "content": "You are a brilliant game designer"},
                {"role": "user", "content": prompt},]
    first_description = prompt_completion_chat(messages=messages, n=1, temperature=0.0, max_tokens=3800, model_id=args.llm_model)

    messages.append({"role": "assistant", "content": f"{first_description}"})
    messages.append({"role": "user", "content": f"""That's great! I want to edit this down a bit though. Can you narrow it down to a shorter list of your top 6 favorite existing mechanics for this set theme? These should be balanced and supportive of the draft archetypes we want. Explain how each mechanic works. Describe what makes the mechanic fun to play and draft.
    
Then, rewrite the 10 color-pair draft archetypes. Connect them to the mechanics you chose. Describe what each archetype is like to draft. What makes a card good for this archetype? Describe what makes the archetype fun to play.

Remember, the set theme is: {args.set_description}

Then write one paragraph giving thematic guidance for this set."""},)
    better_description = prompt_completion_chat(messages=messages, n=1, temperature=0.2, max_tokens=1000, model_id=args.llm_model)

    print("Here's the description we got:")
    print(better_description)

    log_generation_step("set description", prompt, better_description, args.set_name)

    return better_description


def generate_story_and_elements(args):
    print(f"Generating story and elements for {args.set_name}")
    prompt = f"""I'm creating cards for Magic the Gathering, and I want to start by creating a rich story and using that to guide the cards that I might want to create.

The theme of the set is: {args.set_description}

# Story

First, please write out the story of the setting. If the theme is historical or mythological, please draw heavily from the source material. If the theme is original, please write a story that is consistent with the theme. Be sure to mention the roles of common people and more ordinary participants, including lesser monsters, in addition to the the story of the heroes. Try to include an even mixture of men and women characters. Please write at least 500 words, but feel free to write more if you want to.

# Elements

Looking at the story, I'd like to identify possible cards. A card could represent one of these things:
* A named character
* A background character or group, like the people of a village or something like that
* An artifact or object
* A location
* A monster or creature
* An event or important action
* Significant cultural or magical phenomena

# List of Elements

Please list {args.set_size * 2} elements that you think would make good cards. Please include a fair number of less exciting elements, because we need a lot of commons.

Write the name of each one on a separate line starting with a *. """
    messages = [{"role": "system", "content": "You are a game designer who loves stories but also good mechanics"},
                {"role": "user", "content": prompt}, ]

    story_and_elements = prompt_completion_chat(messages=messages, n=1, temperature=0.2, max_tokens=3200,
                                                model_id=args.llm_model)

    print("Story and elements generated.")
    print(story_and_elements)

    log_generation_step("story and elements", prompt, story_and_elements, args.set_name)

    suggested_elements = []
    for line in story_and_elements.split("\n"):
        if line.strip().startswith("*"):
            suggested_elements.append(remove_bullet_etc(line).replace(".", ""))  # Periods mess up parsing

    suggested_elements_as_str = "\n".join([f"* {element}" for element in suggested_elements])

    prompt = f"""I have this list of potential ideas for magic cards:
                                                       
{suggested_elements_as_str}

For each element we've suggested, please suggest the card type such as creature or artifact, rarity, and color or colors that you think would make the most sense to it. 

Try to approximately balance the colors. Include an even mix of commons, uncommons, and rares. Each card should have a unique and interesting name.

I also want you to rate the subjective coolness of the card on a scale from 1-10 where 5 is about average. Coolness is a combination of its importance to the story and its suitability as a Magic card.

Like this:

* Card Name. Card Type. Rarity. Color. Coolness X.

Each line of your response must be exactly in this format! Do not include any other details or commentary!
* Card Name. Card Type. Rarity. Color. Coolness X."""
    described_cards = prompt_completion_chat(messages=[{"role": "system", "content": "You are a game designer who loves stories but also good mechanics"},
                                                       {"role": "user", "content": prompt}], n=1, temperature=0.2, max_tokens=3200, model_id=args.llm_model)

    print("Described cards generated.")
    print(described_cards)
    described_suggested_elements = []
    for line in described_cards.split("\n"):
        if line.strip().startswith("*"):
            described_suggested_elements.append(line[line.index("*") + 1:].strip())

    log_generation_step("described cards", prompt, story_and_elements, args.set_name)

    story = ""
    for line in story_and_elements.split("\n"):
        if line.strip().startswith("# List"):
            break  # Story time is over
        if line.strip().startswith("#") or line.strip().startswith("*") or line.strip().startswith("- "):
            continue
        story += line + "\n"
    story = story.strip().replace("\n\n", "\n")

    return story, described_suggested_elements


def generate_card_suggestions(args, num_cards_to_generate: int):
    # TODO(andrew) If the story is already available in story.txt, load that instead of generating it
    story, described_suggested_elements = generate_story_and_elements(args)

    # TODO(andrew) If there are already card suggestions in card_suggestions.txt, load those and incorporate them into the set balancing
    balanced_suggestions = create_balanced_set(described_suggested_elements, args.set_size, args)

    suggestions = []

    # GPT4 has trouble with long lists, so we break it up into chunks
    for i in range(0, len(balanced_suggestions), 10):
        balanced_suggestions_subset = balanced_suggestions[i:i + 10]
        balanced_suggestions_str = "\n".join([f"* {card}" for card in balanced_suggestions_subset])

        messages = [{"role": "system", "content": "You are a game designer who loves flavor and good themes"},
                    {"role": "user", "content": f"""I'm creating cards for a new Magic the Gathering set. 

The idea for the set is: {args.set_description}

# Story
         
{story}

# Cards
                
I have this list of {len(balanced_suggestions_subset)} card ideas:

{balanced_suggestions_str} 

For each card, I want you to describe its place in the story with a brief description of what it is most known for. 

You may want to rewrite the names of some cards to make them more exciting, thematic and unique.

Write each card on its own line, like this:

* Card Name. Card Type. Rarity. Color. Thematic description of the card.

Please don't describe the full stats of each card, just give a few words to help shape the theme of the card."""},]

        card_suggestions = prompt_completion_chat(messages=messages, n=1, temperature=0.2, max_tokens=1000, model_id=args.llm_model)

        print(f"Here are the cards we got:")
        print(card_suggestions)

        log_generation_step("card suggestions", messages[1]["content"], card_suggestions, args.set_name)

        # These look like cards I guess
        for line in card_suggestions.split("\n"):
            if line.startswith("-") or line.startswith("—") or line.startswith("*") or line.startswith("•") or line.startswith(">") or line.startswith("•"):
                suggestions.append(f"{line}")

    return suggestions, story
