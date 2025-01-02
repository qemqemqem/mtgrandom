import random
import re
from typing import Tuple

from content_utils.gpt import prompt_completion_chat
from set_logging.logger import log_generation_step

with open('content_utils/artists_for_inspiration.txt') as f:
    artists_for_inspiration = [l for l in f.read().splitlines() if l.strip() != ""]
with open('content_utils/art_styles_for_inspiration.txt') as f:
    art_styles_for_inspiration = [l for l in f.read().splitlines() if l.strip() != ""]


def get_art_prompt(card, args) -> Tuple[str, str]:
    name = card['name']
    flavor = card['flavor'] if 'flavor' in card else "I'm not sure"
    mechanics = card['text'] if 'text' in card else "I'm not sure"
    type_line = card['type'] if 'type' in card else "I'm not sure"

    inspiration = [random.choice(artists_for_inspiration), random.choice(artists_for_inspiration), random.choice(art_styles_for_inspiration), random.choice(art_styles_for_inspiration)]

    art_guidance = "\n\n" + args.art_guidance if args.art_guidance else ""

    prompt = f"""I'm generating art for a Magic the Gathering card. 

The details of the card are:
Name: {name}
Flavor Text: {flavor}
Mechanics: {mechanics}
Type: {type_line}

I'd like help generating a prompt to use to create art. First, I'd like you to brainstorm some options, and then I'd like you to write a prompt. 

The final prompt needs to be relatively short, so we'll brainstorm first and then distill it into a short prompt. {art_guidance}

Please fill out this form:

# Brainstorming

Central Figure: Describe what the card might feature
Character Details: If the card features a person, list some details about them. What ethnicity are they? What are they wearing? What is their expression?
Source Material: If this card is based on an existing story or media, what is it?
Action: What action or event could we depict on the card? Ideally this should suggest tension or arouse the viewer's curiosity
Background Details: What is the setting of the scene? What is the lighting like? What is the weather like? 
Culture: What cultural influences are present? If it is not a traditional Western culture, how can we prompt the artist to depict it in a way that is not stereotypical?
Style: List several stylistic influences we might choose
Medium: List several words to describe the visual medium
Fantasy Artist: List several artists, living or dead, who might be appropriate to emulate for this art. 
Historical Artist: This is part of our "Historical Greats" collection, where we feature some of the greatest artists throughout history, to bring their work to a new audience. Consider drawing on an artist from a different culture or underrepresented group. 

(Some of the other cards are in the style of: {', '.join(inspiration)})

# Decision

From the options that you've brainstormed, loosely describe the card art. Try to think of a way in which the scene will be exciting or memorable. Don't be afraid to make bold visual choices to ensure a beautiful card. How can the description convey the cultural influences in a distinctive way? {art_guidance}

# Prompt

Final Prompt: \"[Card Name], from [Source], [describe the scene in one sentence or less], [list several adjectives], in the style of [style], [visual medium], by [artist name] and [historical artist]\"

Artist Credit: [artist name], [second artist, if more than one]"""

    # TODO(andrew): Experiment with removing "Magic the Gathering art, spec art," from the prompt

    temperature = 0.0

    # Try several times with increasing temperature
    for _ in range(3):
        response = prompt_completion_chat(messages=[{"role": "system", "content": "You are an artistic assistant who works with Magic the Gathering"}, {"role": "user", "content": prompt}], n=1, temperature=temperature, max_tokens=2048, model=args.llm_model)

        for line in response.split("\n"):
            if line.startswith("Final Prompt:") or "Magic the Gathering art, spec art," in line:
                found_prompt = line.replace("Final Prompt:", "").replace("\"", "").strip()
                print("Found a prompt!", found_prompt)

                # Get the artist for use in attribution
                artist = get_artist_name(response, found_prompt)

                log_generation_step("art prompt", "Create an art prompt for this card",
                                    f"Prompt: {found_prompt}\n\nArtist: {artist}\n\nFull: {response}", args.set_name if args else None,
                                    name)

                return found_prompt, artist

        temperature += 0.3

    raise ValueError("Art direction prompt not found")


def get_artist_name(full_response, final_prompt):
    if "Artist Credit:" in full_response:
        artist = full_response.split("Artist Credit:")[1].split("\n")[0].strip()
        return artist

    # If the artist is not specified, try to find it in the prompt
    for attribution in [r"by", r"By", r"style of"]:
        artist_pattern = r"\b" + attribution + r"\b\s+([A-Z][^\s]*(?:\s+[A-Z][^\s]*)*)"
        artist = re.findall(artist_pattern, final_prompt)
        if artist:
            # print(f"Artist found: {artist[0]}")
            return artist[0]
    lazy_artist_pattern = r"\b([A-Z][^\s]*(?:\s+[A-Z][^\s]*)*)"
    if re.findall(lazy_artist_pattern, final_prompt):
        artist = re.findall(lazy_artist_pattern, final_prompt)[0]
        # print(f"Artist found: {artist}")
        return artist
    print("Artist not found in prompt:", final_prompt)
    return "Unknown"


if __name__ == "__main__":
    # Sample description strings
    found_prompts = [
        "A cool bird drawn by Kehinde Wiley",
        "An abstract painting by Zdzisław Beksiński",
        "A surreal artwork by H.R. Giger",
        "a cool wizard in the style of Picasso",
        "nice graffiti of a squid that was done by Banksy",
        "cool picture of a mountain",
    ]

    for found_prompt in found_prompts:
        artist = get_artist_name(found_prompt, found_prompt)
        print(f"Artist: {artist}")
