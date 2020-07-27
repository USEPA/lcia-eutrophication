# https://medium.com/better-programming/how-to-flatten-a-dictionary-with-nested-lists-and-dictionaries-in-python-524fd236365


def flatten(d, sep="_"):
    import collections

    obj = collections.OrderedDict()

    def recurse(t, parent_key=""):

        if isinstance(t, list):
            for i in range(len(t)):
                recurse(t[i], parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)

    return obj


# ## TEST EXECUTION BELOW ##
# from pprint import pprint
#
# data = {
#     "id": 1,
#     "first_name": "Jonathan",
#     "last_name": "Hsu",
#     "employment_history": [
#         {
#             "company": "Black Belt Academy",
#             "title": "Instructor",
#             "something": {
#                 "hello": [1, 2, 3, {
#                     "something": "goes"
#                 }]
#             }
#         },
#         {
#             "company": "Zerion Software",
#             "title": "Solutions Engineer"
#         }
#     ],
#     "education": {
#         "bachelors": "Information Technology",
#         "masters": "Applied Information Technology",
#         "phd": "Higher Education"
#     }
# }
#
# pprint(flatten(data), indent=2)


