import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


def _annotate_frame(frame_group, frame_idx, total_frames, frame_name, hide_angles=False):
    frame = np.array(frame_group['image'])

    if frame.ndim == 2:
        frame_display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        frame_display = frame.copy()

    offset_s = frame_group.attrs.get('offset_s', 0)
    inference_ms = frame_group.attrs.get('inference_ms', 0)
    has_detection = frame_group.attrs.get('has_detection', 0)
    
    cv2.putText(
        frame_display,
        f"Frame {frame_idx}/{total_frames} | Time: {offset_s:.2f}s | Inference: {inference_ms:.1f}ms | {frame_name}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    if has_detection:
        try:
            h, w = frame_display.shape[:2]
            
            # Get single box
            if 'box' in frame_group:
                box = np.array(frame_group['box'])
                if len(box) >= 4:
                    x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
                    x1 = max(0, min(x1, w))
                    x2 = max(0, min(x2, w))
                    y1 = max(0, min(y1, h))
                    y2 = max(0, min(y2, h))

                    if x2 > x1 and y2 > y1:
                        cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 0), 3)

                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        cv2.circle(frame_display, (center_x, center_y), 5, (0, 0, 255), -1)

                        confidence = frame_group.attrs.get('confidence', 0.0)
                        azimuth = frame_group.attrs.get('azimuth', 0.0)
                        elevation = frame_group.attrs.get('elevation', 0.0)

                        if not hide_angles:
                            cv2.putText(
                                frame_display,
                                f"Conf: {confidence:.2f} | Az: {azimuth:.1f}deg El: {elevation:.1f}deg",
                                (x1, max(15, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (255, 255, 0),
                                1,
                            )

            cv2.putText(
                frame_display,
                "Detection Found",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        except Exception as e:
            print(f"Error drawing detections: {e}")
            cv2.putText(
                frame_display,
                f"Error: {str(e)[:40]}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
    else:
        cv2.putText(
            frame_display,
            "No detection",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )

    return frame_display


def visualize_hdf5_log(h5_path, save_vid=False, output_video=None, fps=10.0, hide_angles=False):
    """
    Visualize HDF5 log file with frame navigation.

    Controls:
    - 'n': Next frame
    - 'p': Previous frame
    - 'q': Quit
    - Arrow keys: Jump 10 frames
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        print(f"Error: {h5_path} not found")
        return

    with h5py.File(h5_path, 'r') as h5_file:
        frames_group = h5_file['frames']
        frame_names = sorted(frames_group.keys())
        total_frames = len(frame_names)

        if total_frames == 0:
            print("No frames found in HDF5 file")
            return

        print(f"Total frames: {total_frames}")

        if save_vid:
            output_path = Path(output_video) if output_video else h5_path.with_name(f"{h5_path.stem}_annotated.mp4")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            writer = None
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            for frame_idx, frame_name in enumerate(frame_names):
                frame_group = frames_group[frame_name]
                frame_display = _annotate_frame(frame_group, frame_idx, total_frames, frame_name, hide_angles=hide_angles)

                if writer is None:
                    height, width = frame_display.shape[:2]
                    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open video writer for {output_path}")

                writer.write(frame_display)

            if writer is not None:
                writer.release()

            print(f"Saved annotated video to: {output_path}")
            return

        print("Controls: 'n' next | 'p' prev | 'q' quit | Arrow keys jump 10\n")

        frame_idx = 0

        while True:
            frame_name = frame_names[frame_idx]
            frame_group = frames_group[frame_name]
            frame_display = _annotate_frame(frame_group, frame_idx, total_frames, frame_name, hide_angles=hide_angles)

            cv2.imshow("HDF5 Visualization", frame_display)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            if key == ord('n'):
                frame_idx = (frame_idx + 1) % total_frames
            elif key == ord('p'):
                frame_idx = (frame_idx - 1) % total_frames
            elif key == 82:
                frame_idx = max(0, frame_idx - 10)
            elif key == 84:
                frame_idx = min(total_frames - 1, frame_idx + 10)
            elif key == 81:
                frame_idx = max(0, frame_idx - 1)
            elif key == 83:
                frame_idx = min(total_frames - 1, frame_idx + 1)

    cv2.destroyAllWindows()
    print("Visualization closed.")


def parse_opt():
    parser = argparse.ArgumentParser(description="Visualize HDF5 log files from RT-DETR detection.")
    parser.add_argument(
        "--log-file",
        type=str,
        default="/home/jetsonnano/Documents/cv_pipeline_v1/RT-DETR/files/log_hdf5/annotated_frames.h5",
        help="Path to HDF5 log file",
    )
    parser.add_argument(
        "--save-vid",
        action="store_true",
        help="Save annotated frames to a video instead of displaying them interactively",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Output video path used with --save-vid; defaults to <log stem>_annotated.mp4",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Video FPS used when saving with --save-vid",
    )
    parser.add_argument(
        "--hide-angles",
        action="store_true",
        help="Do not draw azimuth/elevation text on frames when saving",
    )
    return parser.parse_args()


if __name__ == "__main__":
    opt = parse_opt()
    visualize_hdf5_log(opt.log_file, save_vid=opt.save_vid, output_video=opt.output_video, fps=opt.fps, hide_angles=opt.hide_angles)