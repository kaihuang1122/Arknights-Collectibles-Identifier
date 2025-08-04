import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import json
import argparse


def start_match():
    global img, kp_img, des_img, flann, sift
    img = cv2.imread(args.img_path)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img
    sift = cv2.SIFT_create()
    kp_img, des_img = sift.detectAndCompute(gray, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=12)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

def match_one(template_path: str, min_match_count: int):
    tpl = cv2.imread(template_path)#, cv2.IMREAD_GRAYSCALE)
    kp_tpl, des_tpl = sift.detectAndCompute(tpl, None)
    if des_tpl is None:
        return False

    matches = flann.knnMatch(des_tpl, des_img, k=2)
    good_matches = [m for m,n in matches if m.distance < 0.7 * n.distance]
    if len(good_matches):
        pts_tpl = np.float32([kp_tpl[m.queryIdx].pt for m in good_matches]).reshape(-1,2)
        pts_img = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1,2)
        M, mask = cv2.estimateAffinePartial2D(
            pts_tpl, pts_img,
            method=cv2.RANSAC,
            ransacReprojThreshold=3,
            maxIters=2000
        )
        if M is None:
            return False
        a, b = M[0,0], M[0,1]
        theta = np.degrees(np.arctan2(b, a))
        if abs(theta) < 1.0:
            inlier_matches = [good for good, m in zip(good_matches, mask.ravel()) if m]
        else:
            inlier_matches = []
    else:
        inlier_matches = []

    # draw points on image
    if SHOW and len(inlier_matches) >= MIN_MATCH_COUNT:
        img_matches = cv2.drawMatches(tpl, kp_tpl, img, kp_img, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow('Matches', img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return len(inlier_matches) >= MIN_MATCH_COUNT


# print(match_one(Path(template_dir)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match collectibles in a screenshot.')
    parser.add_argument('--img_path', type=str, required=True, help='Directory containing collectible templates.')
    parser.add_argument('--IS', type=int, default=6, help='IS version (default: Sui\'s Garden of Grotesqueries)')
    parser.add_argument('--show', action='store_true', help='Show matching results.')
    args = parser.parse_args()
    if args.IS > 6 or args.IS < 1:
        print("IS must be 1 to 6")
        exit(1)
    path = f"./assets/IS{args.IS}_Collectibles.json"
    SHOW = args.show
    img = cv2.imread(args.img_path)
    with open(path, 'r', encoding='utf-8') as f:
        collectibles = json.load(f)
    count = 0
    for item in collectibles:
        item["path"] = Path(item['img_url'][item['img_url'].find("assets/"):])
        if item['name'].find("-") >= 0: # 書，極難分辨
            MIN_MATCH_COUNT = 20
        else:
            MIN_MATCH_COUNT = 7
        if match_one(item["path"], MIN_MATCH_COUNT):
            print(f"{item['name']}")
            count += 1
    print(f"{count} collectibles found in the screenshot.")
