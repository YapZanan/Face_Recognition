import pickle

import face_recognition
import numpy as np
from sklearn import svm
import os

train_dir = 'Training/'


def check_folder():
    return len(os.listdir(train_dir))


def get_images():
    encodings = []
    names = []
    for person in os.listdir(train_dir):
        pix = os.listdir(f"{train_dir}/{person}")
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(f"{train_dir}/{person}/{person_img}")
            face_bounding_boxes = face_recognition.face_locations(face)

            # If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                names.append(person)
            else:
                print(f"{person}/{person_img} was skipped because it contains more than one face or no faces at all")
    return encodings, names


def train(encoding, name):
    clf = svm.SVC(gamma='scale')
    clf.fit(encoding, name)
    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))
    print(type(clf))
    return clf
