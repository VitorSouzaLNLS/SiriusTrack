using SiriusTrack
using SiriusTrack.Auxiliary

model = SiriusTrack.Models.StorageRing.create_accelerator()
model.cavity_state = on
model.vchamber_state = on
model.radiation_state = 1

model