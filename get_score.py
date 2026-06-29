import numpy as np
import pandas as pd
import logging
import cv2
import glob
import os
from helper import cdist, Hungarian
#from scipy.spatial.distance import cdist
#from scipy.optimize import linear_sum_assignment

PENALTY_N_WRONG_HEXBUGS = -100  # only applies if the amount of predicted hexbugs is between 1 and 4
PENALTY_WRONG_NUMBER_FRAMES = -10000
PENALTY_FALSE_ID = -50
#Rings in Pixel
RINGS=          [1,  5 ,10,15,20,25]
POINTS_PER_RING=[100,50,25,15,10,5]
STREAK_POINTS= [0,20,40,60,200]
MAXIMUM_HEXBUGS=11
MAGIC_NUMBER =50
def get_score_fct(path_to_prediction: str, path_to_gt: str, log: bool = False, vid: bool = False) -> int:
    """
    Calculate the score for the given prediction and ground truth files.
    :param path_to_prediction: Path to the prediction .csv file.
    :param path_to_gt: Path to the ground truth .csv file.
    :param log: Boolean to indicate if the log should be written to a file.
    :return: Score for the given prediction and ground truth files.
    """
    final_score = 0

    if log:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(path_to_prediction.replace(".csv", "_log.log"))
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

    # Load the data
    pred_df = pd.read_csv(path_to_prediction, index_col=0)
    gt_df = pd.read_csv(path_to_gt, index_col=0)
    #print(gt_df)
    #print(pred_df)
    # check if the predictions dataframe has at least one row
    if pred_df.shape[0] == 0:
        raise ValueError("The prediction DataFrame does not contain any rows.")

    # Check if DataFrames have the required columns
    required_columns = ['t', 'hexbug', 'x', 'y']
    if not _validate_dataframe_structure(pred_df, required_columns):
        raise ValueError(f"The prediction DataFrame does not have the required columns: {required_columns}")
    else:
        pass
    # Check if any value in the 'hexbug' column is greater than 10.
    # This has to do that the ids_for_streak is only 10 long. So only ids till maxnumber_hexbugs are safed
    if (pred_df['hexbug'] > MAXIMUM_HEXBUGS).any():
        if log:
            logger.info(f"Your ids should not be greater than 10!")
        return -10000
    # Get the number of frames and check if they are equal
    n_frames_gt = gt_df['t'].max() + 1
    n_frames_pred = pred_df['t'].max() + 1

    if n_frames_gt != n_frames_pred:
        final_score += PENALTY_WRONG_NUMBER_FRAMES * np.abs(n_frames_gt - n_frames_pred)

        if log:
            logger.info(f"Penalty for wrong number of frames: "
                        f"{PENALTY_WRONG_NUMBER_FRAMES * np.abs(n_frames_gt - n_frames_pred)}")

        # set n_frames depending on which one is smaller
        n_frames = min(n_frames_gt, n_frames_pred)
        #do not make video if frames are incorrect
        vid = False
    else:
        n_frames = n_frames_gt

    # OpenCV VideoWriter object to write the video
    if vid:
        video_input_path = path_to_gt.replace(".csv", ".mp4")
        output_video_path = path_to_prediction.replace(".csv", "_video.mp4")
        input_video  = cv2.VideoCapture(video_input_path)
        # Check if camera opened successfully
        if (input_video .isOpened() == False):
            print("Error opening video stream or file")
        width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #print(width,height)
        fps = int(input_video.get(cv2.CAP_PROP_FPS))
        output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    #initialize dict to store the ids and streak data of the hexbugs
    #[id, strak, framesofstrak] for n hexbugs
    ids_for_streak = {i: [MAGIC_NUMBER, False, 0] for i in range(MAXIMUM_HEXBUGS)}# Maximum of 10 hexbugs?
    # Loop over all frames
    for idx in range(n_frames):
        if log:
            logger.info(f"Frame {idx}")

        # Get the data for the current frame
        frame_gt_df = gt_df[gt_df['t'] == idx]
        #Look if that frame is even availabe in the prediction
        if(pred_df[pred_df['t'] == idx].empty):
            final_score += PENALTY_FALSE_ID
            if log:
                logger.info(f"Penalty because frame does not exist: "
                            f"{PENALTY_N_WRONG_HEXBUGS*2}")
            continue
        frame_pred_df = pred_df[pred_df['t'] == idx]
        # Check the number of hexbugs and apply penalties if necessary
        n_hexbugs_gt = frame_gt_df['hexbug'].nunique()
        n_hexbugs_pred = frame_pred_df['hexbug'].nunique()
        #used to connect ids with the entries of the distance matrix
        ids_gt = frame_gt_df['hexbug']
        ids_pred = frame_pred_df['hexbug']
        #look if there are ids doubled
        duplicate_values = ids_pred.duplicated()
        if(duplicate_values.any()):
            #print(duplicate_values)
            final_score += -2000
            if log:
                logger.info(f"Penalty for assigning the same id to multible hexbug: "
                            f"{-2000}. "
                            f"This frame is skipped.")
            continue

        #If there is not the same number of hexbug as in gt_frame
        if(np.abs(n_hexbugs_pred - n_hexbugs_gt) != 0):
            final_score += (np.abs(n_hexbugs_pred - n_hexbugs_gt) * PENALTY_N_WRONG_HEXBUGS)
            if log:
                logger.info(f"Penalty for wrong number of hexbugs: "
                            f"{np.abs(n_hexbugs_pred - n_hexbugs_gt) * PENALTY_N_WRONG_HEXBUGS}")
        # Checking for NaN values in columns 'x' and 'y'
        if (frame_pred_df[['x', 'y']].isna().any().any()):
            final_score += PENALTY_FALSE_ID
            if log:
                logger.info(f"Penalty for NaNs in Dataframe: "
                            f"{PENALTY_N_WRONG_HEXBUGS}")
            continue
        # Calculate the distance between the hexbugs
        #here is in the website x and y changed because test data is inverted
        distance_matrix = cdist(list(frame_gt_df[['x', 'y']].values),list(frame_pred_df[['x', 'y']].values), 'euclidean')
        # get hexbugs with shortest distance
        #print(distance_matrix)
        hungarian = Hungarian(distance_matrix)
        hungarian.calculate()
        results = hungarian.get_results()
        gt_hex = [t[0] for t in results]
        pred_hex = [t[1] for t in results]
        #for debugging
        # print(ids_for_streak)
        # print(gt_hex)
        # print(pred_hex)
        #print(gt_hex,pred_hex)

        #make video
        if vid:
            ret, frame = input_video.read()
            if not ret:
                break
            # Process the frame (draw a dot)
            processed_frame = plot_pred_and_gt(list(frame_gt_df[['x', 'y']].values), list(frame_pred_df[['x', 'y']].values), frame,gt_hex, pred_hex,idx)
            # cv2.imshow("Frame with Dot", processed_frame)
            # cv2.waitKey(0)
            output_video.write(processed_frame)

        for i, j in zip(gt_hex, pred_hex):
            # i is the gt hex and j the pred hex with the shortest distance between them
            # get ids of the hex i and hex j
            gt_id = ids_gt.iloc[i]
            pred_id = ids_pred.iloc[j]
            #look if hex is on streak
            #print(i,j,ids_for_streak[gt_id][0],pred_id)
            if(ids_for_streak[gt_id][0] == pred_id):
                ids_for_streak[gt_id][1] = True # the streak is on
                ids_for_streak[gt_id][2] += 1
            #if not on streak then give penalty and reset streak
            else:
                #give penalty if id is used by any other hexbug
                keys_with_50 = [key for key, entry in ids_for_streak.items() if key != gt_id and entry[0] == pred_id]
                if(keys_with_50):
                    ids_for_streak[keys_with_50[0]][0] = 50
                    final_score += PENALTY_FALSE_ID
                    if log:
                        logger.info(f"Penalty for wrong ID for hexbug: {gt_id}"
                                    f". Id {pred_id} was already assigned to hexbug {keys_with_50[0]} earlier "
                                    f"and now is assigned to hexbug {gt_id}."
                                    f" Penalty : {PENALTY_FALSE_ID}")
                #give penalty of not first frame or a new hexbug
                if(ids_for_streak[gt_id][0] != MAGIC_NUMBER):
                    final_score += PENALTY_FALSE_ID
                    if log:
                        logger.info(f"Penalty for wrong ID for hexbug: {gt_id} "
                                    f"which was hexbug {ids_for_streak[gt_id][0]} "
                                    f"and now is hexbug {pred_id}."
                                    f"Penalty : {PENALTY_FALSE_ID}")
                #give hexbugs new id after penalty or if this is first frame
                ids_for_streak[gt_id][0] = pred_id
                ids_for_streak[gt_id][1] = False  # the streak is of
                ids_for_streak[gt_id][2] = 0

            distance = distance_matrix[i, j]
            points_recived = 1
            #Score per Ring in which the prediction is
            for intervalls in range(len(RINGS) - 1):
                if RINGS[intervalls] < distance <= RINGS[intervalls + 1]:
                    points_recived = POINTS_PER_RING[intervalls+1]
                    break
                if(distance <= 1):
                    points_recived = 100

            #Look which streak
            multiplikator = 0
            for intervall in range(len(STREAK_POINTS) - 1):
                if STREAK_POINTS[intervall] < ids_for_streak[i][2] <= STREAK_POINTS[intervall + 1]:
                    multiplikator = (intervall+ 1)
                    points_recived = points_recived * multiplikator
                    break

            final_score += points_recived

            if log:
                logger.info(f"Distance between hexbug {gt_id} and "
                            f"{pred_id}: {distance_matrix[i, j]}   "
                            f"Points recived : {points_recived}   "
                            f" Hexbug on Fire: {ids_for_streak[gt_id][1]} with multiplicator x{multiplikator}")

        if log:
            logger.info(f"Score: {final_score}")
            logger.info("")

    if vid:
        input_video.release()
        output_video.release()
        cv2.destroyAllWindows()

    if log:
        logger.info(f"\nFinal score: {final_score}")
        logger.removeHandler(handler)
        handler.close()
    #One should not get minuspoints for a video. Max is zero
    if(final_score < 0 ):
        return 0
    else:
        return final_score

def plot_pred_and_gt(gt_coord, pred_coord, frame, gt_hex_ids, pred_hex_ids, frame_number):
    # Draw dots on the frame based on the coordinates
    frame = add_legend(frame,frame_number)
    for i in range(len(gt_hex_ids)):
        cv2.circle(frame, gt_coord[i].astype(int),radius= 25, color= (0, 0, 255),thickness=-1)
        cv2.circle(frame, gt_coord[i].astype(int), radius=20, color=(255, 0, 0), thickness=-1)
        cv2.circle(frame, gt_coord[i].astype(int), radius=10, color=(255, 255, 0), thickness=-1)
        cv2.circle(frame, gt_coord[i].astype(int), radius=3, color=(255, 0, 255), thickness=-1)
        cv2.circle(frame, pred_coord[pred_hex_ids[i]].astype(int), radius=3, color=(255, 255, 255), thickness=-1)
        cv2.putText(frame, str(gt_hex_ids[i]), gt_coord[i].astype(int)+10, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3)
        cv2.putText(frame, str(pred_hex_ids[i]), pred_coord[pred_hex_ids[i]].astype(int)+40, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return frame


def add_legend(frame,frame_number):
    # Define legend entries
    legend_entries = [
        ("Ground Truth", (30, 30), (255, 0, 255)),  # Text, position, color
        ("Prediction", (30, 60), (255, 255, 255)),
        ("Frame Number: "+str(frame_number), (30, 90), (255, 0, 0))  # Text, position, color
    ]

    # Draw legend entries on the frame
    for text, position, color in legend_entries:
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    return frame

def _validate_dataframe_structure(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate if a Pandas DataFrame has the required columns.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        required_columns (list): A list of column names that should be present in the DataFrame.

    Returns:
        bool: True if the DataFrame has the required columns, False otherwise.
    """
    # Get the columns present in the DataFrame
    columns = df.columns.tolist()

    # Check if all required columns are present in the DataFrame
    if set(required_columns).issubset(columns):
        return True
    else:
        return False

def run_on_folder(team):
    test_folder = "uploads"
    pred_folder = f"uploads/{team}/{team}"
    #pred_folder = "uploads"

    # Get list of all files in the test folder
    test_files = glob.glob(os.path.join(test_folder, "*.csv"))
    sum =0
    for test_file in test_files:
        # Get the base name of the test file (e.g., test003.csv)
        base_name = os.path.basename(test_file)
        # Construct the corresponding prediction file path
        pred_file = os.path.join(pred_folder, base_name)

        # Check if the corresponding prediction file exists
        if os.path.exists(pred_file):
            score = get_score_fct(pred_file, test_file, log=False, vid=False)
            sum += score
            print(f"Score for {base_name}: {score}")
        else:
            print(f"Prediction file not found for {base_name}")
    print(f"SCORE : ---------{sum}-------")


if __name__ == "__main__":
    #extract args
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("path_to_prediction", help="path to the prediction file")
    # parser.add_argument("path_to_gt", help="path to the ground truth file")
    # parser.add_argument("--log", help="log the score", action="store_true")
    # args = parser.parse_args()
    path_pred = "uploads/HexTrackers/test003.csv"
    path_test = "test/test003.csv"
    teams = ["BugHunters","HacksBug","HexagonHackers","HexTrackers","JustAJoke","PitchCrawlers","StrafeSnipers"]
    #run_on_folder("HexagonHackers")


    #print(f"Score: {get_score_fct(args.path_to_prediction, args.path_to_gt, args.log)}")
    print(f"Score: {get_score_fct(path_pred, path_test, log = True, vid = False)}")

