#!/usr/bin/python3

import pickle

if __name__ == "__main__":
  fr=open('../gtzan.pickle', 'rb')
  root=pickle.load(fr)
  for key, val in root['files'].items():
       for feat_name, feat_value in val['features'].items():
               if feat_name == 'Beat':
                       print("%s: %s" % (key, feat_value['value']))
