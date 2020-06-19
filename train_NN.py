import sys, os
from NNTraining import NNTraining

if len(sys.argv)<3:
    print('Usage:', sys.argv[0], '<batch_size> <path_to_NPZ_file>')
    exit()

bs = int(sys.argv[1])
npz_path = sys.argv[2]

NNT = NNTraining(batch_size=bs, batch_size_valid=bs)

NNT.train_on_npz(npz_path, validation_fraction=0.1)




