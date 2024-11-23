def process_face_recognition_output(face_results, alert_threshold=6):

    '''
    Parameters : 
        face_results (list) : List of face recognition results from the model.
        alert_threshold (int) : Threshold for alerting the user. Default is 6.

    Returns : 
        log : Face N : Detected "NAME" with confidence "X"
    '''

    output = []
    alert_count = 0
    alert_message = "ALERT: Multiple continuous faces unrecognized!"

    for i, result in enumerate(face_results):
        if result["name"] == "Unknown":
            alert_count += 1
        else:
            alert_count = 0  # Reset counter if match is found
        
        if alert_count >= alert_threshold:
            output.append(alert_message)
            alert_count = 0  # Reset after alert

    return "\n".join(output)
