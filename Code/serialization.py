"""
Handles serialization and deserialization to save data samples.
"""
import os
import pickle

def serialize_object(obj, file_name):
  with open(file_name, 'wb') as file_handler:
    pickle.dump(obj, file_handler, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickled(file_name):
  pickled_data = {}
  if os.path.getsize(file_name) > 0:
    with open(file_name, "rb") as f:
      unpickler = pickle.Unpickler(f)
      pickled_data = unpickler.load()
  return pickled_data