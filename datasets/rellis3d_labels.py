"""
# File taken from https://github.com/mcordts/cityscapesScripts/
# License File Available at:
# https://github.com/mcordts/cityscapesScripts/blob/master/license.txt

# ----------------------
# The Cityscapes Dataset
# ----------------------
#
#
# License agreement
# -----------------
#
# This dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:
#
# 1. That the dataset comes "AS IS", without express or implied warranty. Although every effort has been made to ensure accuracy, we (Daimler AG, MPI Informatics, TU Darmstadt) do not accept any responsibility for errors or omissions.
# 2. That you include a reference to the Cityscapes Dataset in any work that makes use of the dataset. For research papers, cite our preferred publication as listed on our website; for other media cite our preferred publication as listed on our website or link to the Cityscapes website.
# 3. That you do not distribute this dataset or modified versions. It is permissible to distribute derivative works in as far as they are abstract representations of this dataset (such as models trained on it or additional annotations that do not directly include any of our data) and do not allow to recover the dataset or something similar in character.
# 4. That you may not use the dataset or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
# 5. That all rights not expressly granted to you are reserved by us (Daimler AG, MPI Informatics, TU Darmstadt).
#
#
# Contact
# -------
#
# Marius Cordts, Mohamed Omran
# www.cityscapes-dataset.net

"""
from collections import namedtuple


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for you approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

# labels = [
#     #       name                     id    trainId   color
#     Label(  'void'                 ,  0 ,      255 , (0, 0, 0) ),
#     Label(  'dirt'                 ,  1 ,      255 , (0, 0, 0) ), #(108, 64, 20) ),
#     Label(  'grass'                ,  3 ,        0 , (0, 102, 0) ),
#     Label(  'tree'                 ,  4 ,        1 , (0, 255, 0) ),
#     Label(  'pole'                 ,  5 ,        2 , (0, 153, 153) ),
#     Label(  'water'                ,  6 ,        3 , (0, 128, 255) ),
#     Label(  'sky'                  ,  7 ,        4 , (0, 0, 255) ),
#     Label(  'vehicle'              ,  8 ,        5 , (255, 255, 0) ),
#     Label(  'object'               ,  9 ,        6 , (255, 0, 127) ),
#     Label(  'asphalt'              , 10 ,        7 , (64, 64, 64) ),
#     Label(  'building'             , 12 ,        8 , (255, 0, 0) ),
#     Label(  'log'                  , 15 ,        9 , (102, 0, 0) ),
#     Label(  'person'               , 17 ,       10 , (204, 153, 255) ),
#     Label(  'fence'                , 18 ,       11 , (102, 0, 204) ),
#     Label(  'bush'                 , 19 ,       12 , (255, 135, 204) ),
#     Label(  'concrete'             , 23 ,       13 , (170, 170, 170) ),
#     Label(  'barrier'              , 27 ,       14 , (41, 121, 255) ),
#     Label(  'puddle'               , 31 ,       15 , (134, 255, 239) ),
#     Label(  'mud'                  , 33 ,       16 , (99, 66, 34) ),
#     Label(  'rubble'               , 34 ,       17 , (110, 22, 138) ),
# ]

labels = [
    #       name                     id    trainId   color
    Label(  'void'                 ,  0 ,        0 , (0, 0, 0) ),
    Label(  'dirt'                 ,  1 ,        0 , (0, 0, 0) ), #(108, 64, 20) ),
    Label(  'grass'                ,  3 ,        1 , (0, 102, 0) ),
    Label(  'tree'                 ,  4 ,        2 , (0, 255, 0) ),
    Label(  'pole'                 ,  5 ,        3 , (0, 153, 153) ),
    Label(  'water'                ,  6 ,        4 , (0, 128, 255) ),
    Label(  'sky'                  ,  7 ,        5 , (0, 0, 255) ),
    Label(  'vehicle'              ,  8 ,        6 , (255, 255, 0) ),
    Label(  'object'               ,  9 ,        7 , (255, 0, 127) ),
    Label(  'asphalt'              , 10 ,        8 , (64, 64, 64) ),
    Label(  'building'             , 12 ,        9 , (255, 0, 0) ),
    Label(  'log'                  , 15 ,       10 , (102, 0, 0) ),
    Label(  'person'               , 17 ,       11 , (204, 153, 255) ),
    Label(  'fence'                , 18 ,       12 , (102, 0, 204) ),
    Label(  'bush'                 , 19 ,       13 , (255, 135, 204) ),
    Label(  'concrete'             , 23 ,       14 , (170, 170, 170) ),
    Label(  'barrier'              , 27 ,       15 , (41, 121, 255) ),
    Label(  'puddle'               , 31 ,       16 , (134, 255, 239) ),
    Label(  'mud'                  , 33 ,       17 , (99, 66, 34) ),
    Label(  'rubble'               , 34 ,       18 , (110, 22, 138) ),
]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------

# Please refer to the main method below for example usages!

# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId      : label for label in reversed(labels) }
# label2trainid
label2trainid   = { label.id      : label.trainId for label in labels        }
# trainId to label object
trainId2name   = { label.trainId       : label.name for label in labels      }
trainId2color  = { label.trainId       : label.color for label in labels     }


#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of RELLIS-3D labels:")
    print("")
    print(("    {:>21} | {:>3} | {:>7}".format( 'name', 'id' , 'trainId')))
    print(("    " + ('-' * 98)))
    for label in labels:
        print(("    {:>21} | {:>3} | {:>7}".format( label.name, label.id, label.trainId)))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'grass'
    id   = name2label[name].id
    print(("ID of label '{name}': {id}".format( name=name, id=id )))

    # Map from ID to label
    label = id2label[id]
    print(("Label with ID '{id}': {label}".format( id=id, label=label.name )))
