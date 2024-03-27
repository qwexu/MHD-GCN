import mne
import numpy


def load_data():

    base_path = 'data/S%03dR%02d.edf'
    subjects = 104
    train_data = numpy.zeros((subjects * 42, 64, 1, 657))
    train_label = numpy.zeros((subjects * 42))

    MI_tasks_2_class = [4,8,12]

    total_id = 0

    for subject_id in range(1, subjects + 6):
        for MI_task in MI_tasks_2_class:
            if subject_id not in [88, 89, 92, 100,104]:
                path = base_path % (subject_id, MI_task)
                print(path)

                raw = mne.io.read_raw_edf(path)

                events,event_dict= mne.events_from_annotations(raw)
                raw.load_data()

                raw.filter(0., 38., fir_design="firwin")
                picks = mne.pick_types(raw.info,meg=False,eeg=True,eog=False,stim=False)

                # Extracts epochs of 4.1s time period from the datset
                tmin,tmax = 0.,4.1

                event_id = dict({'T2':2,'T3':3})
                epochs = mne.Epochs(raw,events,event_id,tmin,tmax,proj=True,picks=picks,baseline=None,preload=True)

                data = epochs.get_data()[0:14] * 1e06

                if MI_task == 4 or MI_task == 8 or MI_task == 12:
                    labels = epochs.events[:,-1] -2
                    labels = labels[0:14]
                else:
                    labels = epochs.events[:,-1]
                    labels = labels[0:14]

                train_data[total_id:total_id + 14] = data.reshape((14, 64, 1,657))
                train_label[total_id:total_id + 14] = labels
                total_id = total_id + 14

    return train_data,train_label
