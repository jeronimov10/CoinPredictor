from configparser import ConfigParser
import os

def config(filename="database.ini", section="postgresql"):

    parser = ConfigParser()
    path = os.path.join(os.path.dirname(__file__), filename)
    parser.read(path)
    
    db = {}

    if parser.has_section(section):
        params = parser.items(section)
        for i in params:
            db[i[0]] = i[1]
    else:
        raise Exception('Section: {0} is not found in the {1} file'.format(section, filename))

    return db
