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
    Label(  ''                     ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  ''                     , 28 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'garden tressel'       , 34 ,        0 , 'void'            , 0       , False        , False        , (  0,  0,  0) ),
    Label(  ''                     , 46 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  ''                     , 50 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'outer wall'           , 53 ,        1 , 'void'            , 0       , False        , False        , (153, 108, 6) ),
    Label(  ''                     , 56 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  ''                     , 59 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'roof'                 , 64 ,        2 , 'void'            , 0       , False        , False        , (112,105,191) ),
    Label(  ''                     , 69 ,      255 , 'flat'            , 1       , False        , True         , (128, 64,128) ),
    Label(  ''                     , 71 ,      255 , 'flat'            , 1       , False        , True         , (128, 64,128) ),
    Label(  'tree'                 , 85 ,        3 , 'flat'            , 1       , False        , False        , ( 89,121, 72) ),
    Label(  'fence'                , 95 ,        4 , 'flat'            , 1       , False        , False        , (190,225, 64) ),
    Label(  'roof'                 , 96 ,        2 , 'flat'            , 1       , False        , False        , (230,150,140) ),
    Label(  'roof'                 , 99 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'outer wall'           ,101 ,        1 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'water'                ,104 ,        5 , 'construction'    , 2       , False        , False        , (206,190, 59) ),
    Label(  'fence'                ,105 ,        4 , 'construction'    , 2       , False        , False        , (180,165,180) ),
    Label(  ''                     ,110 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  ''                     ,113 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  ''                     ,118 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  ''                     ,120 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  ''                     ,121 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'swimming pool'        ,122 ,        6 , 'object'          , 3       , False        , False        , ( 81, 13, 36) ),
    Label(  'roof'                 ,124 ,        2 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  ''                     ,125 ,      255 , 'object'          , 3       , False        , True         , (250,170, 30) ),
    Label(  ''                     ,129 ,      255 , 'object'          , 3       , False        , True         , (250,170, 30) ),
    Label(  ''                     ,132 ,      255 , 'object'          , 3       , False        , True         , (250,170, 30) ),
    Label(  ''                     ,136 ,      255 , 'object'          , 3       , False        , True         , (250,170, 30) ),
    Label(  ''                     ,140 ,      255 , 'object'          , 3       , False        , True         , (220,220,  0) ),
    Label(  'chair'                ,141 ,        7 , 'nature'          , 4       , False        , False        , (115,176,195) ),
    Label(  ''                     ,142 ,      255 , 'object'          , 3       , False        , True         , (220,220,  0) ),
    Label(  'outer wall'           ,144 ,        1 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  ''                     ,145 ,      255 , 'sky'             , 5       , False        , True         , ( 70,130,180) ),
    Label(  'sky'                  ,146 ,        8 , 'human'           , 6       , True         , False        , (161,171, 27) ),
    Label(  ''                     ,147 ,      255 , 'human'           , 6       , True         , True         , (255,  0,  0) ),
    Label(  ''                     ,148 ,      255 , 'object'          , 3       , False        , True         , (220,220,  0) ),
    Label(  ''                     ,151 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,142) ),
    Label(  'leaves'               ,152 ,        9 , 'vehicle'         , 7       , True         , False        , (135,169,180) ),
    Label(  'outer wall'           ,153 ,        1 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  ''                     ,156 ,      255 , 'object'          , 3       , False        , True         , (220,220,  0) ),
    Label(  'street sign'          ,163 ,       10 , 'vehicle'         , 7       , True         , False        , (29, 26, 199) ),
    Label(  ''                     ,164 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'stone road'           ,168 ,       11 , 'vehicle'         , 7       , True         , False        , (102, 16,239) ),
    Label(  'outer wall'           ,170 ,        1 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'tree'                 ,171 ,        3 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  ''                     ,173 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  ''                     ,178 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  ''                     ,180 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  ''                     ,181 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  'fence'                ,182 ,        4 , 'vehicle'         , 7       , False        , False        , (  0,  0,142) ),
    Label(  ''                     ,183 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  ''                     ,185 ,      255 , 'object'          , 3       , False        , True         , (220,220,  0) ),
    Label(  'outer wall'           ,189 ,        1 , 'vehicle'         , 7       , False        , False        , (  0,  0,142) ),
    Label(  ''                     ,190 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  'outer wall'           ,194 ,        1 , 'vehicle'         , 7       , False        , False        , (  0,  0,142) ),
    Label(  'tree'                 ,196 ,        3 , 'vehicle'         , 7       , False        , False        , (  0,  0,142) ),
    Label(  ''                     ,198 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  ''                     ,199 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  ''                     ,200 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  ''                     ,201 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  ''                     ,203 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  'grass'                ,205 ,       12 , 'vehicle'         , 7       , False        , False        , (242,107,146) ),
    Label(  'outer wall'           ,206 ,        1 , 'vehicle'         , 7       , False        , False        , (  0,  0,142) ),
    Label(  ''                     ,208 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  ''                     ,209 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  ''                     ,210 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  'outer wall'           ,217 ,        1 , 'vehicle'         , 7       , False        , False        , (  0,  0,142) ),
    Label(  'roof'                 ,218 ,        2 , 'vehicle'         , 7       , False        , False        , (  0,  0,142) ),
    Label(  'car'                  ,220 ,       13 , 'vehicle'         , 7       , False        , False        , (156,198, 23) ),
    Label(  'pole'                 ,221 ,       14 , 'vehicle'         , 7       , False        , False        , ( 49, 89,160) ),
    Label(  'roof'                 ,222 ,        2 , 'vehicle'         , 7       , False        , False        , (  0,  0,142) ),
    Label(  'outer wall'           ,223 ,        1 , 'vehicle'         , 7       , False        , False        , (  0,  0,142) ),
    Label(  'hedge'                ,224 ,       15 , 'vehicle'         , 7       , False        , False        , ( 68,218,116) ),
    Label(  'fence'                ,227 ,        4 , 'vehicle'         , 7       , False        , False        , (  0,  0,142) ),
    Label(  ''                     ,228 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  ''                     ,229 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  'roof'                 ,231 ,        2 , 'vehicle'         , 7       , False        , False        , (  0,  0,142) ),
    Label(  ''                     ,232 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  'outdoor wall'         ,233 ,        1 , 'vehicle'         , 7       , False        , False        , (  0,  0,142) ),
    Label(  'roof'                 ,236 ,        2 , 'vehicle'         , 7       , False        , False        , (  0,  0,142) ),
    Label(  'outdoor wall'         ,237 ,        1 , 'vehicle'         , 7       , False        , False        , (  0,  0,142) ),
    Label(  ''                     ,239 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  ''                     ,241 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  ''                     ,242 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  ''                     ,245 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  ''                     ,247 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  ''                     ,248 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  'driveway'             ,251 ,       16 , 'vehicle'         , 7       , False        , False        , ( 11,236,  9) ),
    Label(  ''                     ,252 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  'tree'                 ,253 ,        3 , 'vehicle'         , 7       , False        , False        , (  0,  0,142) ),
    Label(  ''                     ,254 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    Label(  ''                     ,255 ,      255 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
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
#label2trainid   = { label.id      : label.trainId for label in labels   }
label2trainid = {i : 255 for i in range(256)}
for label in labels:
    label2trainid[label.id] = label.trainId
#print(label2trainid)
# trainId to label object
trainId2name   = { label.trainId : label.name for label in labels   }
trainId2color  = { label.trainId : label.color for label in labels      }
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
