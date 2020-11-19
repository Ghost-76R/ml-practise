import tensorflow.compat.v1 as tf

sess1=tf.InteractiveSession()
tf.disable_eager_execution()
#init=tf.global_variables_initializer()
x=tf.Variable(10)
y=tf.Variable(20)
z=x*y
sess1.run(x.initializer)
sess1.run(y.initializer)
#print(x.graph is tf.get_default_graph())
print(sess1.run(z))

print(x.graph)
print(tf.get_default_graph())
sess1.close()
"""
#creating new graph
graph2=tf.Graph()
#tf.reset_default_graph()
with graph2.as_default():
    t=tf.Variable(30)
print(t.graph)
print(tf.get_default_graph())
"""
#InteractiveSession() is used initialize a new default session
sess3=tf.InteractiveSession()
print(tf.get_default_session())
print(sess1,'\n',sess3)
sess3.close()
with tf.Session() as sess2:
    sess2.run(x.initializer)
    sess2.run(y.initializer)
    print(z.eval())
