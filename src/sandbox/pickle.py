import pickle

obj = [4, 8, 15, 16, 23, 42]

f = open('store.pckl', 'wb')
pickle.dump(obj, f)
f.close()

f = open('store.pckl', 'rb')
obj = pickle.load(f)
f.close()

print(obj)
