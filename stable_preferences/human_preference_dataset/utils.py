import json
import random


class HumanPreferenceDatasetReader:
    def __init__(self, file_path):
        self.data = self.read_json_file(file_path)
        self.user_hash_index = self._build_user_hash_index()
        self.id_index = self._build_id_index()

    def read_json_file(self, file_path):
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
        return data

    def _build_user_hash_index(self):
        user_hash_index = {}
        for item in self.data:
            user_hash = item["user_hash"]
            if user_hash not in user_hash_index:
                user_hash_index[user_hash] = []
            user_hash_index[user_hash].append(item)
        return user_hash_index

    def _build_id_index(self):
        id_index = {}
        for item in self.data:
            id_index[item["id"]] = item
        return id_index

    def get_data_by_user_hash(self, user_hash):
        return self.user_hash_index.get(user_hash, [])

    def get_data_by_id(self, id):
        return self.id_index.get(id, None)

    def get_all_ids(self):
        return list(self.id_index.keys())

    def get_all_paths(self):
        paths = []
        for item in self.data:
            paths.extend(item["file_path"])
        return paths

    def get_all_user_hashes(self):
        return list(self.user_hash_index.keys())

    def get_paths_by_user_hash(self, user_hash):
        data = self.get_data_by_user_hash(user_hash)
        paths = []
        for item in data:
            paths.extend(item["file_path"])
        return paths

    def get_ids_by_user_hash(self, user_hash):
        data = self.get_data_by_user_hash(user_hash)
        return [item["id"] for item in data]

    def get_prompts_by_user_hash(self, user_hash):
        data = self.get_data_by_user_hash(user_hash)
        return [item["prompt"] for item in data]

    def get_all_prompts(self):
        return [item["prompt"] for item in self.data]

    def sample_random_prompt(self):
        return random.choice(self.get_all_prompts())
