# Does feature scaling affect every machine learning algorithm?

The goal isn't the accuracy number itself, but understanding why some models move and others don't. 

## Question
 
#### Does every machine learning model require feature scaling?
 
Going in, my gut answer was "probably not, but I've never actually watched it happen."
Most tutorials just tell you to `StandardScaler()` everything by default, so I wanted to
isolate the effect on same data, same split, same models and  scaling as the only variable and see who actually moves.