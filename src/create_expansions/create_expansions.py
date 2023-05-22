from transformers import pipeline
import pandas as pd
import sys
import pickle
import argparse
import os

parser = argparse.ArgumentParser(description="SAMPLE DESCRIPTIONS USING LLMs")
parser.add_argument("--start_idx", default=0, type=int, help="start index")
args = parser.parse_args()

start_idx = args.start_idx
print("start_idx", start_idx)

generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B", device=0)

cc_image_labels_tsv = pd.read_csv(
    "/dccstor/leonidka1/data/cc3m_LLM_outputs/image_labels.tsv", sep="\t", header=None
)
df = cc_image_labels_tsv.iloc[start_idx : start_idx + 1000]

# missing_cc3m_idx = pickle.load( open( "missing_cc3m_idx.p", "rb" ) )
# df = cc_image_labels_tsv.iloc[missing_cc3m_idx[start_idx:start_idx+20000]]

for index, row in df.iterrows():
    file_path = (
        f"/dccstor/leonidka1/data/cc3m_LLM_outputs/GPT_NEO_LIST_DESC/{index}.txt"
    )
    if os.path.isfile(file_path):
        print(f"file already exists {file_path}")
        continue
    short_description = row[0]
    # input_text = f"short description: Please describe what we see in a scene of christmas tree on a black background.\n \
    # long description: In a scene of a Christmas tree on a black background, one would see a decorated tree in the center of the \
    # image, lit up with lights and possibly adorned with ornaments and a star or angel at the top. The background would \
    # be dark and black in color. The tree is decorated with colorful lights and ornaments, with a bright star on top or \
    # an angel topper. The tree is standing on the black background, which creates a contrast and makes the tree stand out. \n \
    # short description: Please describe what we see in a scene of male hand playing nervously with a pencil on a black background. \n \
    # long description: In a scene of a male hand playing nervously with a pencil on a black background, you might see the following:\n\n* \
    # A male hand: This could be a hand belonging to a person of any age, with fingers that are long or short, thin or thick, \
    # and with skin that is pale or dark. The hand might be moving quickly or slowly, and it might be fidgeting or making jerky \
    # movements.\n* A pencil: The pencil might be any color or length, and it might be made of wood or another material. \
    # It might be sharpened or unsharpened, and it might have an eraser on one end.\n* A black background: This could be \
    # a solid black color, or it could be a darker shade of another color. The background might be featureless or textured, \
    # and it might be completely flat or slightly curved.\n\nTogether, these elements would create a scene in which the male \
    # hand is playing nervously with the pencil on a black background. The overall mood of the scene might be tense or anxious, \
    # depending on the movements and expressions of the hand and the context in which the scene is set\n \
    # short description: Please describe what we see in a scene of a man is programming\n \
    # long description: In a scene of a man programming, you might see the following:\n\n* A man: This could be a person of any age, \
    # with any physical appearance or style of dress. He might be seated at a desk or table, or he might be standing or \
    # walking around.\n* A computer: The computer could be a desktop or a laptop, and it might be running any operating \
    # system or software. The screen of the computer might be displaying code, text, images, or other types of data.\n* \
    # Other equipment: The man might be using other equipment or tools in addition to the computer, such as a mouse, k\
    # eyboard, or additional monitors. He might also have other objects on his desk, such as papers, books, or pens.\n\n\
    # Overall, the scene would depict the man working on programming tasks, using a computer and potentially other equipment \
    # to create, test, and debug code. The man might be focused and intense as he works, or he might be relaxed and casual. \
    # The specific context and details of the scene will depend on the particular setting and the tasks the man is working on.\n \
    # short description: Please describe what we see in a scene of {short_description}\n \
    # long description: "
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

    res = generator(
        input_text,
        do_sample=True,
        max_length=len(input_text.split()) + 800,
        min_length=len(input_text.split()) + 600,
    )
    long_description_out = res[0]["generated_text"].replace(input_text, "")

    f = open(
        f"/dccstor/leonidka1/data/cc3m_LLM_outputs/GPT_NEO_LIST_DESC/{index}.txt", "w"
    )
    f.write(f"{short_description}\n")
    f.write(f"{long_description_out}\n")
    f.close()

    # print (index, long_description_out)
