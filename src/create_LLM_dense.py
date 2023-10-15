from transformers import pipeline
import json
import os
import glob
os.umask(0)

if __name__ == '__main__':

    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B', device=0)
    os.makedirs("../LLM_dense/", exist_ok=True)
    for json_file_name in glob.glob("../quality_captions/*.json"):
        clean_json_file_name = os.path.basename(json_file_name)[:-5]
        file_path = f"../LLM_dense/{clean_json_file_name}.txt"
        if os.path.isfile(file_path):
            print(f"file already exists {file_path}")
            continue
        # read JSON file
        with open(json_file_name) as f:
            data = json.load(f)
        short_description = data['positive_caption'][0]
        input_text = f"short: please describe what you might see in a picture of a scene that contains 'a Christmas tree', write each sentence in a list, and use complete sentences with all nouns and objects you are referring to\n \
                long: 	1	In the center of the room, a majestic evergreen Christmas tree stands tall, adorned with twinkling lights and colorful ornaments.\n \
        	2	Delicate strands of tinsel gracefully drape the tree's branches, adding a touch of shimmer to the festive display.\n \
        	3	An elegant star or angel graces the top of the tree, representing the Star of Bethlehem or the heavenly messengers present at Jesus' birth.\n \
        	4	Wrapped presents in various shapes and sizes are piled beneath the tree, their festive gift wrap and bows hinting at the surprises inside.\n \
        	5	A cozy fireplace crackles nearby, with stockings hung from the mantel, eagerly awaiting the arrival of Santa Claus.\n \
        	6	Lush green garlands and flickering candles decorate the mantel, enhancing the holiday atmosphere.\n \
        	7	Comfortable seating arrangements, such as sofas and armchairs, are positioned near the tree, complete with plush cushions and warm throw blankets.\n \
        	8	Family members and friends gather around the tree in festive attire, sharing laughter and conversation.\n \
        	9	A beautifully crafted wreath hangs on a nearby wall or window, adding an additional touch of holiday cheer.\n \
        	10	Through the window, a snowy winter landscape can be seen, with snow-covered trees, rooftops, and gently falling snowflakes, creating the perfect backdrop for the Christmas scene.\n \
                short: please describe what you might see in a picture of a scene that contains 'a male hand playing nervously with a pencil on a black background', write each sentence in a list, and use complete sentences with all nouns and objects you are referring to \n \
                long:   1	A male hand is positioned prominently in the frame, with fingers flexing and shifting as they manipulate a pencil.\n \
        	2	The pencil, held between the thumb and index finger, twirls and spins as the hand moves it nervously.\n \
        	3	Shadows from the hand and pencil cast dramatic patterns on the stark black background, emphasizing the sense of tension and unease.\n \
        	4	Flecks of graphite from the pencil's tip may be visible, scattered across the black surface, as a result of the restless movements.\n \
        	5	The hand's knuckles and veins are accentuated by the lighting, highlighting the pressure and force exerted during the fidgeting.\n \
        	6	The pencil's eraser end, worn and discolored, suggests frequent use and a history of anxious behavior.\n \
        	7	A hint of perspiration on the hand's skin glistens under the light, further revealing the nervous energy within the scene.\n \
        	8	The positioning of the hand, perhaps slightly off-center or at an angle, contributes to the visual tension of the composition.\n \
        	9	Fingernails on the hand may appear bitten or worn, indicating a habit of nervousness and stress.\n \
        	10	The black background contrasts sharply with the hand and pencil, isolating them in the scene and focusing the viewer's attention on the restless, uneasy motion.\n \
                short: please describe what you might see in a picture of a scene that contains 'a man is programming', write each sentence in a list, and use complete sentences with all nouns and objects you are referring to \n \
                long:   1	A focused man sits at a desk, his eyes intently scanning the computer screen in front of him as he works on a programming project.\n \
        	2	The computer display is filled with lines of code, featuring various colors and syntax highlighting to differentiate between elements of the programming language.\n \
        	3	The man's fingers move swiftly and confidently across the keyboard, typing commands and adjusting the code as needed.\n \
        	4	Beside the keyboard, a mouse and a notepad with handwritten notes or algorithms offer additional tools for the programmer's work.\n \
        	5	A cup of coffee or tea sits nearby, providing the man with a source of caffeine to maintain his focus and energy.\n \
        	6	The room's lighting, either from a desk lamp or overhead lights, illuminates the workspace, creating a comfortable environment for concentration.\n \
        	7	The man wears casual attire, such as a t-shirt and jeans, reflecting the informal nature of the programming process.\n \
        	8	Reference books or technical manuals may be stacked or spread out on the desk, offering guidance and information for the programmer.\n \
        	9	The man's facial expression, furrowed brows or a slight frown, conveys his deep concentration and determination to solve the coding challenge at hand.\n \
        	10	Surrounding the man, other electronic devices, like a smartphone or tablet, may be present, indicating the interconnected nature of his work in the digital realm.\n \
                short: please describe what you might see in a picture of a scene that contains '{short_description}', write each sentence in a list, and use complete sentences with all nouns and objects you are referring to \n \
                long: "

        res = generator(input_text, do_sample=True, max_length=len(input_text.split()) + 800,
                        min_length=len(input_text.split()) + 600)
        long_description_out = res[0]['generated_text'].replace(input_text, "")

        f = open(
            file_path,
            "w")
        f.write(f"{long_description_out}\n")
        f.close()





