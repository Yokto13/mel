class DamuelParser:
    @classmethod
    def parse_description(cls, damuel_entry):
        if "wiki" in damuel_entry:
            return damuel_entry["wiki"]["text"]
        elif "description" in damuel_entry:
            return damuel_entry["description"]
        return None

    @classmethod
    def parse_label(cls, damuel_entry):
        if "label" in damuel_entry:
            return damuel_entry["label"]
        elif "wiki" in damuel_entry:
            return damuel_entry["wiki"]["title"]
        return None

    @classmethod
    def parse_qid(cls, damuel_entry):
        return int(damuel_entry["qid"][1:])
