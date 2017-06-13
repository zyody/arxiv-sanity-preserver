# standard imports
import os
import sys
import pickle
# non-standard imports
import numpy as np
import operator 
from sqlite3 import dbapi2 as sqlite3
# local imports
from recommend_utils import safe_pickle_dump, strip_version, Config

# -----------------------------------------------------------------------------
# database, later be changed to mysql
if not os.path.isfile(Config.database_path):
  print("the database file as.db should exist. You can create an empty database with sqlite3 as.db < schema.sql")
  sys.exit()

sqldb = sqlite3.connect(Config.database_path)
sqldb.row_factory = sqlite3.Row # to return dicts rather than tuples

def query_db(query, args=(), one=False):
  """Queries the database and returns a list of dictionaries."""
  cur = sqldb.execute(query, args)
  rv = cur.fetchall()
  return (rv[0] if rv else None) if one else rv

# -----------------------------------------------------------------------------

# svm_score and cf_score are calculated offline. Now they are assigned from pickle file, later can be from the database.
svm_score = {}
if os.path.isfile(Config.svm_score_path):
  svm_score = pickle.load(open(Config.svm_score_path, 'rb')) # can be negative
else:
  print('generate cache for svm_score first')
  sys.exit()
#sorted_svm_score = [(k, svm_score[2][k]) for k in sorted(svm_score[2], key=svm_score[2].get, reverse=True)]
#print(sorted_svm_score)  # test code

cf_score = {}
if os.path.isfile(Config.cf_score_path):
  cf_score = pickle.load(open(Config.cf_score_path, 'rb'))
else:
  print('generate cache for cf_score first')
  sys.exit()

time_score = {}
if os.path.isfile(Config.time_score_path):
  time_score = pickle.load(open(Config.time_score_path, 'rb'))
else:
  print('generate cache for time_score first')
  sys.exit()

# now type_weight are assigned from pickle file, later can be from the database
type_weight = {}
if os.path.isfile(Config.type_weight_path):
  type_weight = pickle.load(open(Config.type_weight_path, 'rb'))
else:
  print('generate cache for type_weight first')
  sys.exit()

type_weight_update = type_weight# we only modify type_weight_update and mantain type_weight, so that the recommendation is the same in this run

# define different types of sub-models, later can be appended to contain sub-models considering author information, papers from paperweekly, followees, topics of concern etc.
def score_from_svm(uid=None):#already calculated offline
  uid_svm_score = {}
  if uid in svm_score:
    uid_svm_score = svm_score[uid]
  return uid_svm_score

def score_from_cf(uid=None):# already calculated offline
  uid_cf_score = {}
  if uid in cf_score:
    uid_cf_score = cf_score[uid]
  return uid_cf_score

def score_from_time():# add bias to the recommendation list based on published time
  return time_score

def score_from_followees(uid=None):# implemented later when followee is available
  uid_followees = []# replaced by get_followees(uid)
  uid_followees_score = {}
  for uid_followee in uid_followees:
    uid_followee_score = {}# replaced by get_followee_score()
    uid_followees_score[followee] = uid_followee_score
  return uid_followees_score

def mix_filter(uid=None):# mix the candidate papers and filter those unreasonable (e.g. already in library)
  uid_svm_score = score_from_svm(uid)
  uid_cf_score = score_from_cf(uid)
  time_score = score_from_time()
  uid_followees_score = score_from_followees(uid)
  mix_paper_set = set(uid_svm_score.keys()) | set(uid_cf_score.keys()) | set(time_score.keys()) # e.g. select some sub-models or based on proportion
  for uid_followee_score in uid_followees_score:
    mix_paper_set |= set(uid_followee_score.keys())
  lib = query_db('''select * from library where user_id = ?''', [uid])
  filter_set = [x['paper_id'] for x in lib] # filter papers already in library, may add other filter strategies
  paper_set = [x for x in mix_paper_set if x not in filter_set]
  return (paper_set, uid_svm_score, uid_cf_score, time_score, uid_followees_score)

# get weights of different types of scores for a certain user
def get_weight(uid=None):
  uid_weight = {}
  if uid in type_weight:
    uid_weight = type_weight[uid]
  return uid_weight

# a user has a new type of sub model, register a key for it in type_weight[uid], the weight is initialized with 1.
def add_new_type(uid, type_of_submodel):
  if type_of_submodel in type_weight_update[uid]:
    print(type_of_submodel, 'has already in', type_weight_update[uid])
  else:
    uid_followees = []# replaced by get_followees(uid)
    if type_of_submodel in uid_followees:
      type_weight_update[uid][type_of_submodel] = 1#set empirically
    safe_pickle_dump(type_weight_update, Config.type_weight_path)

# a new user set up, register a key for him/her in type_weight 
def add_new_user(uid):
  if uid in type_weight_update:
    print(uid, 'has already in', type_weight_updte)
  else:
    type_weight_update[uid] = {}
    safe_pickle_dump(type_weight_update, Config.type_weight_path)

def update_weight(uid, paper):
  (paper_set, uid_svm_score, uid_cf_score, time_score, uid_followees_score) = mix_filter(uid)
  uid_weight = get_weight(uid)
  if paper in uid_svm_score and 'svm' in uid_weight:
    type_weight_update[uid]['svm'] *= 1.1# simply multiply 1.1 for the click recommendation, you may multiply a larger value if it is collected.
  else:
    type_weight_update[uid]['svm'] *= 0.95# simply multiply 0.9 for the unclick (and uncollected) recommendations
  if paper in uid_cf_score and 'cf' in uid_weight:
    type_weight_update[uid]['cf'] *= 1.1# simply multiply 1.1 for the click recommendation, you may multiply a larger value if it is collected.
  else:
    type_weight_update[uid]['cf'] *= 0.95# simply multiply 0.9 for the unclick (and uncollected) recommendations
  if paper[:4] == '1705' and 'time' in uid_weight:
    type_weight_update[uid]['time'] *= 1.1# simply multiply 1.1 for the click recommendation, you may multiply a larger value if it is collected.
  else:
    type_weight_update[uid]['time'] *= 0.95# simply multiply 0.9 for the unclick (and uncollected) recommendations
  for followee, followee_score in uid_followees_score.items():
    if paper in followee_score and followee in uid_weight:
      type_weight_update[uid][followee] *= 1.1# simply multiply 1.1 for the click recommendation, you may multiply a larger value if it is collected.
    else:
      type_weight_update[uid][followee] *= 0.95# simply multiply 0.9 for the unclick (and uncollected) recommendations
  print(type_weight_update[uid])
  safe_pickle_dump(type_weight_update, Config.type_weight_path)
  # you may use other update strategies. e.g. SGD

def gen_recommend(uid=None, topN = 1000):
  (paper_set, uid_svm_score, uid_cf_score, time_score, uid_followees_score) = mix_filter(uid)
  uid_weight = get_weight(uid)

  #rerank paper list
  paper_score = {}
  for paper in paper_set:
    paper_score[paper] = 0
    if paper in uid_svm_score and 'svm' in uid_weight:
      paper_score[paper] += uid_svm_score[paper]*uid_weight['svm']
    if paper in uid_cf_score and 'cf' in uid_weight:
      paper_score[paper] += uid_cf_score[paper]*uid_weight['svm']
    if paper in time_score and 'time' in uid_weight:
      paper_score[paper] += time_score[paper]*uid_weight['time']
    for followee, followee_score in uid_followees_score.items():
      if paper in followee_score and followee in uid_weight:
        paper_score[paper] += followee_score[paper]*uid_weight[followee]
  sorted_paper_score = sorted(paper_score.items(), key=operator.itemgetter(1), reverse = True)

  #generate explanation
  paper_explanation = []
  for paper,score in sorted_paper_score[:topN]:
    explanation = ''
    is_start = 1
    if paper in uid_svm_score:
      is_start = 0
      explanation += 'recommended based on paper content in your library'
    if paper in uid_cf_score:
      if is_start == 0:
        explanation += ' & '
      else:
        is_start = 0
      explanation += 'users similar to you also read'
    if paper[:4] == '1705':# need to modify, recently published
      if is_start == 0:
        explanation += ' & '
      else:
        is_start = 0
      explanation += 'recently published'
    followees_read = []
    for followee, followee_score in uid_followees_score.items():
      if paper in followee_score:
        followees_read.append(followee)
    if len(followees_read) != 0:
      if is_start == 0:
        explanation += ' & '
      else:
        is_start = 0
      explanation += 'your followee: '
      for followee in followees_read:
        explanation += followee
      explanation += ' also reads'
    explanation += ':'
    paper_explanation.append([paper, explanation])
  return paper_explanation
if __name__ == "__main__":
  paper_explanation = gen_recommend(1)
  print(paper_explanation)
