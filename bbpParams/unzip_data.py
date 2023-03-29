import zipfile as zf
import os
import shutil

for folder in os.listdir('./'):
    if '.' in folder:
        continue
    folderPath = os.path.join('./', folder)
    zipPath = os.path.join(folderPath, 'models.zip')
    with zf.ZipFile(zipPath, 'r') as zip_ref:
        zip_ref.extractall(folderPath)
    os.unlink(zipPath)
    toExtract = list()
    for filename in os.listdir(folderPath):
        if filename.endswith(".zip"): 
            filePath = os.path.join(folderPath, filename)
            os.mkdir(filePath[:-4])
            #print(filePath)
            files = zf.ZipFile(filePath, 'r')
            files.extractall(folderPath)
            files.close()
            filePath = os.path.join(folderPath, filename[:-4])
            for filename2 in os.listdir(filePath):
                filePath2 = os.path.join(filePath, filename2)
                if filename2 == 'biophysics.hoc':
                    continue
                elif os.path.isfile(filePath2):
                    os.unlink(filePath2)
                elif os.path.isdir(filePath2): 
                    shutil.rmtree(filePath2)
    
    for filename in os.listdir(folderPath):
        if filename.endswith(".zip"): 
            os.unlink(os.path.join(folderPath, filename))
