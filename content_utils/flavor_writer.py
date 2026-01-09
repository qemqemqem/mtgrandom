import json
from types import SimpleNamespace

from content_utils.llm import prompt_completion_chat
from set_logging.logger import log_generation_step


def write_flavor_for_card(card_idea, card_json, story, args):
    lines_remaining_for_flavor = 8

    flavorless_json = {k: v for k, v in card_json.items() if k != "flavor_text"}

    for line in card_json["rule_text"].split("\n"):
        lines_in_line = int(len(line) / 50) + 1
        lines_remaining_for_flavor -= lines_in_line

    if lines_remaining_for_flavor > -2:
        lines_remaining_for_flavor = 1  # Eh, let's just always put one line on there unless it's mega cramped

    if lines_remaining_for_flavor < 1:
        print("Not enough room for flavor text, skipping", lines_remaining_for_flavor)
        return ""

    lines_guide = "a single line"
    if lines_remaining_for_flavor > 1:
        lines_guide = f"{lines_remaining_for_flavor} lines"

    len_advice = "keep your ideas short and sweet"
    more_len_advice = ""
    if lines_remaining_for_flavor > 2:
        len_advice = "you can write a paragraph or a poem or a longer quote"
        more_len_advice = f"* Write {lines_remaining_for_flavor} lines of song or verse\n"

    prompt = f"""I'd like help writing the flavor text for this Magic the Gathering card:
    
Here's the idea I had for the card:
    
{card_idea}
    
And here's the card's stats:
    
{card_json}
    
Here's the story of the set:
    
{story}
    
# Brainstorming
    
First, I want you to brainstorm some ideas for the flavor text. Choose 3 of these writing prompts, and write out about {lines_guide} of text for each one:
* A quote attributed to character in the story about the card, like: "A quotation here" -- [Character Name]
* Poetry relevant to the card
* Something humorous
* Reference to another part of the story
* A single line describing a pivotal moment in the story
* A shocking reveal from the story
* A quote from the character that illustrates their personality, like "A quotation here" -- [Character Name]
* A paragraph that narrates a surprising action like betrayal or sacrifice
{more_len_advice}
Because of the rules on the card, we only have about {lines_guide} of text or {lines_remaining_for_flavor * 40} characters to work with. So {len_advice}. Don't use quotation marks unless it's a quote from a character in the story.

# Final Flavor

Now, I want you to think about these options and choose one that best fits the theme and feeling of the card. Write out the flavor text for this card. It should be {lines_guide} of text that evokes the flavor of the card. So {len_advice}.

Please write the flavor text on its own line, like this (no quotation marks unless it's a quote from a character in the story):
    
Flavor: [flavor text here]"""

    flavor_text = prompt_completion_chat(messages=[{"role": "system", "content": "You are a writer for Magic the Gathering cards. You love beautiful prose."}, {"role": "user", "content": prompt}], n=1, temperature=0.2, max_tokens=1000, model_id=args.llm_model)

    final_flavor = ""
    for line in flavor_text.split("\n"):
        if line.startswith("Flavor: "):
            final_flavor = line.replace("Flavor: ", "")
            break
        elif line.strip() != "":
            final_flavor = line  # If they didn't write "Flavor: " at the beginning, just take the last line

    card_name = card_json["name"] if "name" in card_json else "Unknown Card"
    log_generation_step("flavor", "Write flavor text for this card", f"Final flavor: {final_flavor}\n\n{flavor_text}", args.set_name if args else None, card_name)

    print("Final flavor text:", final_flavor)
    return final_flavor


if __name__ == "__main__":
    idea = "Ereshkigal's Minions. Creature. Common. Black. These are the spirits and demons of the netherworld, called upon by Ereshkigal to wage war against the gods and the world of the living."
    card_json = json.loads("""{
    "name": "Ereshkigal's Minions",
    "supertype": "Creature",
    "subtype": "Spirit",
    "power": "2",
    "toughness": "2",
    "rule_text": "Deathtouch\\nWhen this creature dies, each opponent loses 1 life.",
    "flavor_text": "These are the spirits and demons of the netherworld, called upon by Ereshkigal to wage war against the gods and the world of the living.",
    "mana_cost": "{1}{B}",
    "rarity": "Common"
}""")

    story = """For thematic guidance, this set should evoke the grandeur and mystery of ancient Sumerian mythology. The gods should feel powerful and awe-inspiring, with mechanics like devotion and constellation representing their influence and authority. The netherworld should feel eerie and foreboding, with mechanics like embalm and sacrifice representing its dark and dangerous nature. The theme of creation should be represented through the creation of tokens and the growth of creatures, with mechanics like afterlife and devotion. The art and flavor text should further reinforce these themes, drawing from Sumerian myths and iconography to create a rich and immersive world."""

    # args = SimpleNamespace(llm_model=LLMModel.gpt_4, set_name=None)

    flavor = write_flavor_for_card(idea, card_json, story)
