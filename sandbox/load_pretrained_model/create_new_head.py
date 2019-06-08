from project_code.models.networks_head import Head3dmm

head = Head3dmm(199, 'test')
head.build(input_shape=(None, 7, 7, 2048))
print(head.summary())