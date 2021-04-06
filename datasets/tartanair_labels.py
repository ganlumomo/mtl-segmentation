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

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

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

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            , 13 ,        0 , 'void'            , 0       , False        , False        , (156, 198, 23) ),
    Label(  'ego vehicle'          , 64 ,        1 , 'void'            , 0       , False        , False        , (99, 242, 104) ),
    Label(  'rectification border' , 96 ,        2 , 'void'            , 0       , False        , False        , (93, 14, 71) ),
    Label(  'out of roi'           ,110 ,        3 , 'void'            , 0       , False        , False        , (40, 63, 99) ),
    Label(  'static'               ,129 ,        4 , 'void'            , 0       , False        , False        , (108, 116, 224) ),
    Label(  'dynamic'              ,137 ,        5 , 'void'            , 0       , False        , False        , (201, 144, 169) ),
    Label(  'ground'               ,152 ,        6 , 'void'            , 0       , False        , False        , (17, 91, 237) ),
    Label(  'road'                 ,153 ,        7 , 'flat'            , 1       , False        , False        , (31, 95, 84) ),
    Label(  'sidewalk'             ,160 ,        8 , 'flat'            , 1       , False        , False        , (61, 212, 54) ),
    Label(  'parking'              ,163 ,        9 , 'flat'            , 1       , False        , False        , (140, 167, 255) ),
    Label(  'rail track'           ,164 ,       10 , 'flat'            , 1       , False        , False        , (117, 93, 91) ),
    Label(  'building'             ,167 ,       11 , 'construction'    , 2       , False        , False        , (144, 238, 194) ),
    Label(  'wall'                 ,178 ,       12 , 'construction'    , 2       , False        , False        , (138, 223, 226) ),
    Label(  'fence'                ,184 ,       13 , 'construction'    , 2       , False        , False        , (83, 82, 52) ),
    Label(  'guard rail'           ,196 ,       14 , 'construction'    , 2       , False        , False        , (209, 247, 202) ),
    Label(  'bridge'               ,197 ,       15 , 'construction'    , 2       , False        , False        , (178, 221, 213) ),
    Label(  'tunnel'               ,199 ,       16 , 'construction'    , 2       , False        , False        , (244, 117, 51) ),
    Label(  'pole'                 ,200 ,       17 , 'object'          , 3       , False        , False        , (107, 68, 190) ),
    Label(  'polegroup'            ,205 ,       18 , 'object'          , 3       , False        , False        , (105, 127, 176) ),
    Label(  'traffic light'        ,207 ,       19 , 'object'          , 3       , False        , False        , (72, 172, 138) ),
    Label(  'traffic sign'         ,220 ,       20 , 'object'          , 3       , False        , False        , (60, 138, 96) ),
    Label(  'vegetation'           ,222 ,       21 , 'nature'          , 4       , False        , False        , (123, 48, 18) ),
    Label(  'terrain'              ,226 ,       22 , 'nature'          , 4       , False        , False        , (204, 143, 135) ),
    Label(  'sky'                  ,227 ,       23 , 'sky'             , 5       , False        , False        , (249, 79, 73) ),
    Label(  'person'               ,230 ,       24 , 'human'           , 6       , False        , False        , (16, 154, 4) ),
    Label(  'rider'                ,244 ,       25 , 'human'           , 6       , False        , False        , (213, 220, 89) ),
    Label(  'car'                  ,245 ,       26 , 'vehicle'         , 7       , False        , False        , (70, 209, 228) ),
    Label(  'truck'                ,246 ,       27 , 'vehicle'         , 7       , False        , False        , (97, 184, 83) ),
    Label(  'bus'                  ,250 ,       28 , 'vehicle'         , 7       , False        , False        , (38, 27, 159) ),
    Label(  'caravan'              ,252 ,       29 , 'vehicle'         , 7       , False        , False        , (130, 56, 55) ),
    #Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    #Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    #Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    #Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    #Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
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
trainId2label   = { label.trainId : label for label in reversed(labels) }
# label2trainid
label2trainid   = { label.id      : label.trainId for label in labels   }
# trainId to label object
trainId2name   = { label.trainId : label.name for label in labels   }
trainId2color  = { label.trainId : label.color for label in labels  }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]

#--------------------------------------------------------------------------------
# Assure single instance name
#--------------------------------------------------------------------------------

# returns the label name that describes a single instance (if possible)
# e.g.     input     |   output
#        ----------------------
#          car       |   car
#          cargroup  |   car
#          foo       |   None
#          foogroup  |   None
#          skygroup  |   None
def assureSingleInstanceName( name ):
    # if the name is known, it is not a group
    if name in name2label:
        return name
    # test if the name actually denotes a group
    if not name.endswith("group"):
        return None
    # remove group
    name = name[:-len("group")]
    # test if the new name exists
    if not name in name2label:
        return None
    # test if the new name denotes a label that actually has instances
    if not name2label[name].hasInstances:
        return None
    # all good then
    return name

#--------------------------------------------------------------------------------
# Main for testing
#--------------------------------------------------------------------------------

# just a dummy main
if __name__ == "__main__":
    # Print all the labels
    print("List of cityscapes labels:")
    print("")
    print(("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( 'name', 'id', 'trainId', 'category', 'categoryId', 'hasInstances', 'ignoreInEval' )))
    print(("    " + ('-' * 98)))
    for label in labels:
        print(("    {:>21} | {:>3} | {:>7} | {:>14} | {:>10} | {:>12} | {:>12}".format( label.name, label.id, label.trainId, label.category, label.categoryId, label.hasInstances, label.ignoreInEval )))
    print("")

    print("Example usages:")

    # Map from name to label
    name = 'car'
    id   = name2label[name].id
    print(("ID of label '{name}': {id}".format( name=name, id=id )))

    # Map from ID to label
    category = id2label[id].category
    print(("Category of label with ID '{id}': {category}".format( id=id, category=category )))

    # Map from trainID to label
    trainId = 0
    name = trainId2label[trainId].name
    print(("Name of label with trainID '{id}': {name}".format( id=trainId, name=name )))
