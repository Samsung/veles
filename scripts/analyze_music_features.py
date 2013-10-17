#!/usr/bin/python3

import pickle
import numpy

if __name__ == "__main__":
  fr=open('../gtzan.pickle', 'rb')
  root=pickle.load(fr)
  for key, val in root['files'].items():
      for feat_name, feat_value in val['features'].items():
          for i in range(0,len(feat_value['value'])):
              if numpy.isnan(feat_value['value'][i]):
                  print("%s[%d] - %s" % (key, i, feat_name))
