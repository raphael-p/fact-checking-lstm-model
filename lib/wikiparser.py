from tqdm import tqdm
import json


def parseWiki(wikipedia_dir="../../data/wiki-pages/",
               doc_id_dir="../../data/id_to_text.jsonl"):
    """
    This function traverses all the jsonl files
    and returns a dictionary containing document ID and corresponding content
    Args
    wikipedia_dir: the parent directory of the jsonl files
    doc_id_dir: the location of wiki-pages
    Returns
    a dictionary: document ID as dictionary keys and document content as values.

    Remark: Saves the dictionary in ../data/doc_id_text to speed up subsequent passes.
    """
    # doc_id_text saves the title and content of each wiki-page
    doc_id_text = dict()
    try:
        with open(doc_id_dir, "r") as f:
            doc_id = 0
            print("reading from " + str(doc_id_dir) + " ... (/47437710it)")
            for line in tqdm(f):
                fields = line.rstrip("\n").split("\t")
                if (fields[0] == "-1") and fields[1]:
                    if doc_id:
                        doc_id_text[doc_id] = text
                    doc_id = fields[1]
                    text = []
                elif len(fields) > 1:
                    text.append(fields[1])
        f.close()
    except FileNotFoundError:
        with open(doc_id_dir, "w") as w:
            print("\nconstructing " + str(doc_id_dir) + " ...")
            for i in tqdm(range(1, 110)):  # jsonl file number from 001 to 109
                jnum = "{:03d}".format(i)
                fname = wikipedia_dir + "wiki-" + jnum + ".jsonl"
                with open(fname) as f:
                    line = f.readline()
                    while line:
                        data = json.loads(line.rstrip("\n"))
                        doc_id = data["id"]
                        text = data["lines"]
                        if text != "":
                            w.write("-1" + "\t" + doc_id + "\n" + text + "\n")
                            doc_id_text[doc_id] = text
                        line = f.readline()
    return doc_id_text


if __name__ == '__main__':
    wiki_dir = "../data/wiki-pages/"
    doc_id_directory = "../data/id_to_text.jsonl"
    wiki_dict = parseWiki(wiki_dir, doc_id_directory)
    doc_list = ["Nikolaj", "The_Other_Woman_-LRB-2014_film-RRB-", "Nikolaj_Coster-Waldau", "Ved_verdens_ende",
                "Nukaaka_Coster-Waldau", "Jett_Atwood", "List_of_video_game_crowdfunding_projects", "Roman_Atwood",
                "Stack_Overflow", "Payback-COLON-_Debt_and_the_Shadow_Side_of_Wealth",
                "List_of_New_Music_America_performances", "The_Nritarutya_Dance_Collective", "History_of_art", "Kcho",
                "Acropolis_Institute_of_Technology_and_Research", "Adrienne_Bailon",
                "Empire_Girls-COLON-_Julissa_and_Adrienne", "Julissa_Bermudez", "MTV_New_Year's", "All_You've_Got",
                "Homeland_-LRB-TV_series-RRB-", "The_Ten_Commandments_-LRB-1956_film-RRB-", "Tetris", "Cyndi_Lauper",
                "She's_So_Unusual", "The_Hunger_Games_-LRB-film-RRB-"]
    for doc in doc_list:
        print(len(wiki_dict[doc]))
    exit(0)
