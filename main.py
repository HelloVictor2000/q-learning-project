import collections
import random
import time

import numpy as np
import requests


class Game:

    def __init__(self, user_id, api_key, team_id, world_id, q_table_file = None):
        self.user_id = user_id
        self.api_key = api_key
        self.team_id = team_id
        self.base_url_1 = "https://www.notexponential.com/aip2pgaming/api/rl/gw.php"
        self.base_url_2 = "https://www.notexponential.com/aip2pgaming/api/rl/score.php"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "x-api-key": self.api_key,
            "userId": self.user_id
        }

        self.location = [0, 0]
        self.world_id = world_id
        self.world_size = 40
        self.q_table = np.zeros((self.world_size, self.world_size, 4))
        self.epsilon = 0.5
        self.learning_rate = 0.5
        self.discount_factor = 0.9
        self.actions = ['N', 'S', 'E', 'W']

        if q_table_file is not None:
            self.q_table = np.load(q_table_file)

    def api_get_runs(self, count=5):
        params = {
            "type": "runs",
            "teamId": self.team_id,
            "count": count
        }
        response_json = requests.get(self.base_url_2, params=params, headers=self.headers).json()
        if response_json["code"] == "OK":
            return response_json["runs"]


    def api_get_location(self):
        params = {
            "type": "location",
            "teamId": self.team_id
        }
        response_json = requests.get(self.base_url_1, params=params, headers=self.headers).json()
        if response_json["code"] == "OK":
            x, y = response_json["state"].split(':')
            return [int(x), int(y)]

    def api_enter_world(self, world_id):
        data = {
            "type": "enter",
            "worldId": world_id,
            "teamId": self.team_id,
        }
        response_json = requests.post(self.base_url_1, data=data, headers=self.headers).json()
        print(response_json)
        if response_json["code"] == "OK":
            # x, y = response_json["state"].split(':')
            return True
        else:
            return False


    def api_make_move(self, move):
        data = {
            "type": "move",
            "teamId": self.team_id,
            "move": move,
            "worldId": self.world_id
        }
        response_json = requests.post(self.base_url_1, data=data, headers=self.headers).json()
        if response_json["code"] == "OK":
            reward = float(response_json["reward"])
            score_increment = float(response_json["scoreIncrement"])
            if response_json["newState"] is None:
                new_state = None
            else:
                new_state_str = response_json["newState"]
                new_state = [int(new_state_str["x"]), int(new_state_str["y"])]
            return new_state, score_increment, reward

    def api_get_score(self):
        params = {
            "type": "score",
            "teamId": self.team_id,
        }
        response_json = requests.get(self.base_url_2, params=params, headers=self.headers).json()
        if response_json["code"] == "OK":
            return response_json["score"]
        else:
            return None

    def action_index_to_name(self, action_index):
        return self.actions[action_index]

    def q_learning(self):

        start_time = time.time()

        barriers = collections.defaultdict(set)

        # checks if the action doesn't make it go outside the world or go into a visited location:
        def action_valid(action_index):
            if (
                    (self.location[0] == 0 and action_index == 0) or
                    (self.location[0] == 39 and action_index == 1) or
                    (self.location[1] == 0 and action_index == 2) or
                    (self.location[1] == 39 and action_index == 3)
            ):
                return False

            if action_index in barriers[tuple(self.location)]:
                return False

            return True

        while True:

            # choose action:
            while True:
                if random.random() < self.epsilon:
                    action_index = random.randint(0, len(self.actions) - 1)
                    if action_valid(action_index):
                        break
                else:
                    action_index = int(np.argmax(self.q_table[self.location[0], self.location[1]]))
                    break


            # perform action:
            new_location, score_increment, reward = self.api_make_move(self.action_index_to_name(action_index))

            if new_location is None:  # target reached
                target_reward = reward
                last_score_increment = score_increment
                break  # exit the world

            print(f"New location: {new_location} | Reward: {reward}")

            if new_location == self.location:  # this is a barrier
                barriers[tuple(self.location)].add(action_index)  # add to `barriers`

            # update and save the Q-table:
            self.q_table[self.location[0], self.location[1], action_index] += self.learning_rate * (reward + self.discount_factor * np.max(self.q_table[new_location[0], new_location[1]]) - self.q_table[self.location[0], self.location[1], action_index])
            np.save("data.npy", self.q_table)

            self.location = new_location

        score = self.api_get_score()
        print(f"Training on world {self.world_id} finished. Score increment: {last_score_increment}. Reward of last move: {target_reward}. Score: {score}.")
        print("Q-table:")
        print(self.q_table)

        end_time = time.time()

        print(f"Total time of running: {(end_time - start_time):.2f} seconds")

    def run(self):
        self.api_enter_world(self.world_id)
        self.q_learning()


game = Game(
    user_id="",
    api_key="",
    team_id="",
    world_id="",
    q_table_file=None
)
game.run()


