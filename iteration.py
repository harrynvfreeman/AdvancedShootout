import mcts_iteration
from mcts_iteration import best_player_dir, self_play_data_dir
import threading
from multiprocessing import Process
import numpy as np
import time
import os

class Iteration:
    def __init__(self, last_eval=0):
        self.lock = threading.Lock()
        self.evaluating = False
        self.last_eval = last_eval
        self.self_play_thread = None
        self.evaluate_thread = None
        
    def start(self):
        self.self_play_thread = threading.Thread(target=self.self_play)
        self.evaluate_thread = threading.Thread(target=self.evaluate)
        
        self.self_play_thread.start()
        self.evaluate_thread.start()
        
    def self_play(self):
        while True:
            self.lock.acquire()
            best_version = np.load('./train/' + best_player_dir + '/best_version.npy')
            version = np.load('./train/version.npy')
            print('SELF_PLAY: starting selfplay for ' + str(best_version) + ', ' + str(version))
            self.lock.release()
            p = Process(target=mcts_iteration.self_play, args=(best_version, version,))
            p.start()
            p.join()
            self.lock.acquire()
            print('SELF_PLAY: ending selfplay for ' + str(best_version) + ', ' + str(version))
            print('SELF_PLAY: beginning optimize for ' + str(version))
            self.lock.release()
            p = Process(target=mcts_iteration.optimize, args=(version,))
            p.start()
            p.join()
            self.lock.acquire()
            print('SELF_PLAY: ending optimize for ' + str(version))
            version = version + 1
            os.mkdir('./train/' + str(version))
            os.system('mv ./train/opt_tmp/V.npy ./train/' + str(version) + '/V.npy')
            os.system('mv ./train/opt_tmp/P.npy ./train/' + str(version) + '/P.npy')
            np.save('./train/version.npy', version)
            os.mkdir('./train/' + self_play_data_dir + '/' + str(version))
            self.lock.release()
            
    def evaluate(self):
        while True:
            self.lock.acquire()
            best_version = np.load('./train/' + best_player_dir + '/best_version.npy')
            version = np.load('./train/version.npy')
            if version == self.last_eval:
                print('EVALUATE: Evaluate is sleeping because last eval is same as current version: ' + str(version))
                self.lock.release()
                time.sleep(5)
                continue
            print('EVALUATE: starting evaluate for ' + str(best_version) + ', ' + str(version))
            self.last_eval = version
            self.lock.release()
            p = Process(target=mcts_iteration.evaluate, args=(best_version, version,))
            p.start()
            p.join()
            self.lock.acquire()
            print('EVALUATE: ending evaluate for ' + str(best_version) + ', ' + str(version))
            new_best_version = np.load('./train/eval_tmp/best_version.npy')
            if best_version == new_best_version:
                print('EVALUATE: best version not updated')
                self.lock.release()
                continue
            os.mkdir('./train/' + best_player_dir + '/' + str(new_best_version))
            os.system('mv ./train/eval_tmp/P.npy ' './train/' + best_player_dir + '/' + str(new_best_version) + '/P.npy')
            os.system('mv ./train/eval_tmp/V.npy ' './train/' + best_player_dir + '/' + str(new_best_version) + '/V.npy')
            np.save('./train/' + best_player_dir + '/best_version.npy', new_best_version)
            print('EVALUATE: best player updated')
            self.lock.release()

mcts_iteration.initialize()

iteration = Iteration(last_eval=0)
iteration.start()

# i = 0
# while True:
#     print("Starting cycle ", i, " -----------------")
#     i = i + 1
#     mcts_iteration.self_play()
#     mcts_iteration.optimize()
#     if mcts_iteration.evaluate():
#         print('Best player updated')
