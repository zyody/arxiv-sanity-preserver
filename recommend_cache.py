# standard imports
import os
import sys
import pickle
# non-standard imports
import numpy as np
from sklearn import svm
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

# fetch all users
users = query_db('''select * from user''')
print('number of users: ', len(users))
uid_pids = {}
for u in users:
  uid = u['user_id']
  lib = query_db('''select * from library where user_id = ?''', [uid])
  print('''select * from library where user_id = ?''', [uid])
  pids = [x['paper_id'] for x in lib] # raw pids without version
  uid_pids[uid] = pids
print('calculating uid_pids is done')

# load the tfidf matrix
out = pickle.load(open(Config.tfidf_path, 'rb'))
X = out['X']
X = X.todense()
print('loading tfidf matrix is done')

# map from pid to index in tfidf matrix
meta = pickle.load(open(Config.meta_path, 'rb'))
xtoi = { strip_version(x):i for x,i in meta['ptoi'].items() }
print('xtoi', len(xtoi))

def svm_cache(): # the svm submodel, calcaulated offline
  svm_score = {}
  num_of_candidate = 200
  for i,user in enumerate(users):
    print("%d/%d building an SVM for %s" % (i, len(users), user['username'].encode('utf-8')))
    uid = user['user_id']
    pids = uid_pids[uid] # raw pids without version
    posix = [xtoi[p] for p in pids if p in xtoi]
  
    if not posix:
      print('empty library for this user maybe?')
      svm_score[uid] = {}
      continue # empty library for this user maybe?

    y = np.zeros(X.shape[0])
    for ix in posix: y[ix] = 1

    clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=0.1)
    clf.fit(X,y)
    s = clf.decision_function(X)
    sortix = np.argsort(-s)
    sortix_filter = [x for x in sortix if x not in posix] # remove the pids in the library
    max_index = min(num_of_candidate,len(sortix_filter))
    min_value = s[sortix_filter[max_index - 1]]
    max_value = s[sortix_filter[0]]

    # we also need map from index in tfidf matrix to pid
    svm_score[uid] = {strip_version(meta['pids'][ix]):(s[ix] - min_value)/(max_value - min_value) for ix in list(sortix_filter[:max_index])} # crop paper recommendations to save space
  print('writing', Config.svm_score_path)
  safe_pickle_dump(svm_score, Config.svm_score_path)

def cf_cache():# the collaborative filtering submodel
  cf_score = {}
  num_nn = 300# number of nearest number
  for i,user1 in enumerate(users):
    print("%d/%d calculating cf scores for %s" % (i, len(users), user1['username'].encode('utf-8')))
    uid_now = user1['user_id']
    pids_now = uid_pids[uid_now]
    user_sim = {}
    for user2 in users:
      uid = user2['user_id']
      if uid == uid_now:
        continue
      pids = uid_pids[uid]#papers that a certain user saves
      tmp = {x for x in pids if x in pids_now}
      user_sim[uid] = len(tmp)/((len(pids)+1)**0.5 * (len(pids_now)+1)**0.5)#calculate the similarity between user_id and current user, plus 1 to avoid divided by 0
    sorted_user_sim = [(k, user_sim[k]) for k in sorted(user_sim, key=user_sim.get, reverse=True)]#sort uids from the closest to the farthest
    ite = 0
    paper_score = {} 
    for uid,sim_score in sorted_user_sim:
      if sim_score == 0 or ite > num_nn:# all the following sim_score is zero or reach the maximun number
        break
      for pid in uid_pids[uid]:
        if pid in pids_now:
          continue
        if pid not in paper_score:
          paper_score[pid] = 0
        paper_score[pid] += sim_score
      ite += 1
    if paper_score:
      max_score = max(paper_score.values())
    for pid, score in paper_score.items():
      paper_score[pid] = score #/ max_score
    cf_score[uid_now] = paper_score
  print('cf_score:', cf_score)
  print('writing', Config.cf_score_path)
  safe_pickle_dump(cf_score, Config.cf_score_path)

def time_cache():# the submodel considering the publish time. Here I infer the time based on paper id, should be modified to the exact time.
  time_score = {}
  pids_recent = []
  for pid in meta['pids']:
    if pid[:3] == '170' or pid[:3] == '160':
      pids_recent.append(strip_version(pid))
  pids_sort = sorted(pids_recent, reverse = True)
  num_pids = len(pids_sort)
  for i in range(num_pids):
    time_score[pids_sort[i]] = (num_pids - i) / num_pids
  print('writing', Config.time_score_path)
  safe_pickle_dump(time_score, Config.time_score_path)
  
def type_weight_cache():
  type_weight = {}
  if os.path.isfile(Config.type_weight_path):
    type_weight = pickle.load(open(Config.type_weight_path, 'rb'))
  for uid in uid_pids.keys():
    if uid not in type_weight:
      type_weight[uid] = {}
    if 'svm' not in type_weight[uid]:
      type_weight[uid]['svm'] = 1 # set empirically
    if 'cf' not in type_weight[uid]:
      type_weight[uid]['cf'] = 1 # set empirically
    if 'time' not in type_weight[uid]:
      type_weight[uid]['time'] = 0.1 # set empirically
  print('type_weight', type_weight)
  safe_pickle_dump(type_weight, Config.type_weight_path)

if __name__ == "__main__":
  svm_cache()
  cf_cache()
  time_cache()
  type_weight_cache()
