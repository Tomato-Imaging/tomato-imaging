from __future__ import print_function
import os
import numpy as np
import cv2
import re
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])




def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def merger():
    folder = app.config['PROCESSED_FOLDER']

    # We get all the image files from the source folder
    files = list([os.path.join(folder, f) for f in os.listdir(folder)])

    # We compute the average by adding up the images
    # Start from an explicitly set as floating point, in order to force the
    # conversion of the 8-bit values from the images, which would otherwise overflow
    average = cv2.imread(files[1]).astype(np.float)
    for file in files[1:]:
        image = cv2.imread(file)
        # NumPy adds two images element wise, so pixel by pixel / channel by channel
        average += image
 
    # Divide by count (again each pixel/channel is divided)
    average /= len(files)

    # Normalize the image, to spread the pixel intensities across 0..255
    # This will brighten the image without losing information
    output = cv2.normalize(average, None, 0, 255, cv2.NORM_MINMAX)

    # Save the output
    return cv2.imwrite(os.path.join(folder, 'merged.png'), output)

def aligner(im1, im2):
    MAX_MATCHES = 500
    GOOD_MATCH_PERCENT = 0.15
    # We get all the image files from the source folder
    # Convert images to grayscale
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_MATCHES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
    
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    
    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    #imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    #cv2.imwrite("matches.jpg", imMatches)
    
    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt
    
    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))
    
    return im1Reg, h

        

@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    if request.form['submit_button'] == 'Submit':
        if 'photos' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['photos']

        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #print('upload_image filename: ' + filename)
            flash('Image successfully uploaded and displayed')
            return render_template('upload.html', filename=filename)

        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)
    elif request.form['submit_button'] == 'Merge':
            flash('Merging images')
            print('merging images')
            folder = app.config['UPLOAD_FOLDER']
            processed_folder = app.config['PROCESSED_FOLDER']
            # We get all the image files from the source folder
            files = list([os.path.join(folder, f) for f in os.listdir(folder)])
            
            # Read reference image
            refFilename = files[1]
            print("Reading reference image : ", refFilename)

            imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

            # Read image to be aligned
            for file in files[2:]:
                image = cv2.imread(file)
                imFilename = file
                print("Reading image to align : ", imFilename);  
                im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
            
                print("Aligning images ...")
                # Registered image will be resotred in imReg. 
                # The estimated homography will be stored in h. 
                imReg, h = aligner(im, imReference)
            
                # Write aligned image to disk. 
                file = re.findall(r"\d\.jpg", file)
                outFilename = 'aligned' + '-' + str(file[0])
                print("Saving aligned image : ", outFilename); 
                #cv2.imwrite(outFilename, imReg)
                cv2.imwrite(os.path.join(processed_folder, outFilename), imReg)

                # Print estimated homography
                print("Estimated homography : \n",  h)
        
            merger()
            return render_template('upload.html')
    else:
        return render_template('upload.html')

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

