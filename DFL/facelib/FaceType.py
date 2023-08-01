from enum import IntEnum

class FaceType(IntEnum):
    #enumerating in order "next contains prev"
    HALF = 0
    HALF_NO_ALIGN = 30
    MID_FULL = 1
    MID_FULL_NO_ALIGN = 31
    FULL = 2
    FULL_NO_ALIGN = 3
    WHOLE_FACE = 4
    WHOLE_FACE_NO_ALIGN = 32
    HEAD = 10
    HEAD_NO_ALIGN = 20

    MARK_ONLY = 100, #no align at all, just embedded faceinfo

    @staticmethod
    def fromString (s):
        r = from_string_dict.get (s.lower())
        if r is None:
            raise Exception ('FaceType.fromString value error')
        return r

    @staticmethod
    def toString (face_type):
        return to_string_dict[face_type]

to_string_dict = { FaceType.HALF : 'half_face',
                   FaceType.HALF_NO_ALIGN: 'half_face_no_align',
                   FaceType.MID_FULL : 'midfull_face',
                   FaceType.MID_FULL_NO_ALIGN: 'midfull_face_no_align',
                   FaceType.FULL : 'full_face',
                   FaceType.FULL_NO_ALIGN : 'full_face_no_align',
                   FaceType.WHOLE_FACE : 'whole_face',
                   FaceType.WHOLE_FACE_NO_ALIGN : 'whole_face_no_align',
                   FaceType.HEAD : 'head',
                   FaceType.HEAD_NO_ALIGN : 'head_no_align',
                   
                   FaceType.MARK_ONLY :'mark_only',  
                 }

from_string_dict = { to_string_dict[x] : x for x in to_string_dict.keys() }  