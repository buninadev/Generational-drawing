import numpy as np, cv2
import os


class ExportToVideo:
    def __init__(self, inputfolder, outputfolder="videos"):
        self.outputfolder = outputfolder
        self.inputfolder = inputfolder
        if not os.path.exists(self.outputfolder):
            os.mkdir(self.outputfolder)
        self.image_sequences = self.get_image_sequences()
        self.number_of_frames = 0
        self.fps = 30

    def get_image_sequences(self):
        # get all folders in the input folder
        folders = sorted(
            os.listdir(self.inputfolder), key=lambda x: int(x.split("_")[1])
        )
        image_sequences = []

        for folder in folders:
            # get all images in the folder
            sequence_files = [
                f
                for f in os.listdir(os.path.join(self.inputfolder, folder))
                if os.path.isfile(os.path.join(self.inputfolder, folder, f))
            ]
            sequence_files.sort()  # sort the images by name
            image_sequences.append((folder, sequence_files))

        return image_sequences

    def export_all(self, fps):
        self.fps = fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        max_width, max_height = self.get_max_shape_from_sequences(
            self.image_sequences[0][0], self.image_sequences[0][1]
        )
        video = cv2.VideoWriter(
            os.path.join(self.outputfolder, "output.mp4"),
            fourcc,
            fps,
            (max_width, max_height),
        )
        i = 10
        for image_sequence in self.image_sequences:
            folder: str = image_sequence[0]
            files: list = image_sequence[1]
            text_frame = np.zeros((max_height, max_width, 3), np.uint8)
            text = folder.replace("_", " ").upper()
            text_frame = self.write_text_in_center(text_frame, text)
            self.write_frame_to_video(video, text_frame, 8)
            # Write the frames to the video
            for file in sorted(files, reverse=True):
                frame = cv2.imread(os.path.join(self.inputfolder, folder, file))
                self.write_frame_to_video(video, frame, i)
            if i > 1:
                i -= 1
            # Release the video
        video.release()

        return self

    def write_text_in_center(self, text_frame, text):
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 10, 10)[0]
        textX = (text_frame.shape[1] - text_size[0]) // 2
        textY = (text_frame.shape[0] + text_size[1]) // 2
        cv2.putText(
            img=text_frame,
            text=text,
            org=(textX, textY),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=10,
            color=(255, 255, 255),
            thickness=10,
        )
        return text_frame

    def write_frame_to_video(self, video, frame, number_of_frames=1):
        if number_of_frames == 0:
            return
        if number_of_frames == 1:
            video.write(frame)
            self.number_of_frames += 1
            return
        for _ in range(number_of_frames):
            video.write(frame)
            if number_of_frames == 100:
                cv2.imwrite("videos/text_frame.png", frame)
            self.number_of_frames += 1

    def get_max_shape_from_sequences(self, folder, image_sequence):
        max_width = 0
        max_height = 0
        for image in image_sequence:
            img = cv2.imread(os.path.join(self.inputfolder, folder, image))
            if img.shape[0] > max_height:
                max_height = img.shape[0]
            if img.shape[1] > max_width:
                max_width = img.shape[1]
        return (max_width, max_height)

    def count_number_of_images(self):
        count = 0
        for image_sequence in self.image_sequences:
            folder: str = image_sequence[0]
            files = image_sequence[1]
            count += len(files)
        return count


if __name__ == "__main__":
    inputfolder = "incubator_output"
    print(f"Exporting all images in", inputfolder, "to a video")
    export = ExportToVideo(inputfolder).export_all(15)
    print(
        export.number_of_frames,
        " have been exported to a video",
        export.number_of_frames / export.fps,
        " seconds long",
    )
