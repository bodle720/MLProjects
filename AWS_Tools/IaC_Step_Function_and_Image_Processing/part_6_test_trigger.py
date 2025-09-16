# -*- coding: utf-8 -*-
"""
Step function (state machines)

It will use a copy of the hello world exampl i used, called hello-world
"""

from helpers.s3_helpers import upload_local_file_to_s3

path_to_local_img = "stone_building.jpg"

upload_success = upload_local_file_to_s3(path_to_local_img, BUCKET_NAME, object_name = "imgs/stone_building.jpg")
print('Was upload a success?: ', 'Yes!' if upload_success else 'No.')




