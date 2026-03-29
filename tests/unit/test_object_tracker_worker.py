import pytest
import queue
from unittest.mock import MagicMock, patch

from modules.object_tracker.object_tracker_worker import object_tracker_read_loop

class StopLoopException(Exception):
    pass

def test_worker_read_loop_data_flow():
    """Verify the worker loop correctly moves data from the hardware queue to the output queue."""
    mock_device = MagicMock()
    mock_hw_queue = MagicMock()
    mock_device.getOutputQueue.return_value = mock_hw_queue
    
    mock_tracklets_data = MagicMock()
    mock_hw_queue.get.side_effect = [mock_tracklets_data, StopLoopException("Intentional stop")]
    
    out_queue = queue.Queue()
    
    # THE FIX: Create mock TrackedObjects that have the attributes the logger expects
    mock_obj_1 = MagicMock()
    mock_obj_1.object_id = 1
    mock_obj_1.status.value = "TRACKED"
    
    mock_obj_2 = MagicMock()
    mock_obj_2.object_id = 2
    mock_obj_2.status.value = "NEW"
    
    mock_parsed_objects = [mock_obj_1, mock_obj_2]
    
    with patch('modules.object_tracker.object_tracker_worker.parse_tracklets') as mock_parse:
        mock_parse.return_value = mock_parsed_objects
        
        with pytest.raises(StopLoopException):
            object_tracker_read_loop(
                device=mock_device,
                label_map=["person", "car"],
                frame_width=1920,
                frame_height=1080,
                output_queue=out_queue
            )
            
        mock_device.getOutputQueue.assert_called_with(name="tracklets", maxSize=4, blocking=False)
        assert mock_hw_queue.get.call_count == 2
        mock_parse.assert_called_once_with(
            tracklets_data=mock_tracklets_data,
            label_map=["person", "car"],
            frame_width=1920,
            frame_height=1080
        )
        
        assert out_queue.qsize() == 1
        queued_item = out_queue.get_nowait()
        assert queued_item == mock_parsed_objects