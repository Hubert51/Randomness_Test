
class Card(object):
    def __init__(self, number):
        """
            Three attributes:
            Value is 1->A, 2->2, ... 10->10, 11->Jack, 12->Queen, 13-> King
            Type is 1->Heart, 2->Spade, 3->Diamond, 4->Club
            point is A->4, King->3, Queen->2, Jack->1
        """
        self.value = number % 13 + 1
        self.type = number // 13 + 1

        if self.value == 1:
            self.point = 4
        elif self.value == 13:
            self.point = 3
        elif self.value == 12:
            self.point = 2
        elif self.value == 11:
            self.point = 1
        else:
            self.point = 0


    def __repr__(self):
        four_type = ["Heart", 'Spade', 'Diamond', 'Club']
        return "{} {}".format(four_type[self.type-1], self.value)


"""
classdef Card < matlab.mixin.CustomDisplay
    # Card Summary of this class goes here
    # this is the first line 
    # Detailed explanation goes here
    # CARD 123

    properties

    end

    methods
        function self = Card(index1)
            #CARD Construct an instance of this class
            #   Detailed explanation goes here

            self.type = idivide(int32(index1),13) + 1;
            if self.value == 1
                self.point = 4;
            elseif self.value == 13
                self.point = 3;
            elseif self.value == 12
                self.point = 2;
            elseif self.value == 11
                self.point = 1;
            else
                self.point = 0;
            end
            if self.type == 1
                result = 'Heart ';
            elseif self.type == 2
                result = 'Spade ';
            elseif self.type == 3
                result = 'Diamond ';
            elseif self.type == 4
                result = 'Club ';
            end

            result = strcat(result, string(self.value));
            self.repr = result;

        end




        function outputArg = method1(self,inputArg)
            #METHOD1 Summary of this method goes here
            #   Detailed explanation goes here
            outputArg = self.Property1 + inputArg;
        end

        function show(self)
            if self.type == 1
                result = 'Heart ';
            elseif self.type == 2
                result = 'Spade ';
            elseif self.type == 3
                result = 'Diamond ';
            elseif self.type == 4
                result = 'Club ';
            end
            result = strcat(result, string(self.value));
            # disp(result)

        end

        function output = string(self)
            output = self.repr;
        end

    end

    methods (Access = protected)

        function displayScalarObject(self)
            # Implement the custom display for scalar self
            disp(self.repr)
        end

        function displayNonScalarObject(objAry)
            dimStr = matlab.mixin.CustomDisplay.convertDimensionsToString(objAry);
            cName = matlab.mixin.CustomDisplay.getClassNameForHeader(objAry);
            headerStr = [dimStr,' ',cName,' members:'];
            header = sprintf('#s\n',headerStr);
            disp(header);
#             disp(find([objAry.type] == 1));
#             index = [objAry.type] == 1;
#             disp(objAry(index).value);
#             heart = table.hand1( find( [table.hand1.type]==1) );
#             disp(heart)
            output.heart = '';
            output.Spade = '';
            output.Diamond = '';
            output.Club = '';
            for ix = 1:length(objAry)
                o = objAry(ix);
                if o.type == 1
                    output.heart = [output.heart, num2str(o.value), '  '];
                elseif o.type == 2
                    output.Spade = [output.Spade, num2str(o.value), '  '];
                elseif o.type == 3
                    output.Diamond = [output.Diamond, num2str(o.value), '  '];
                elseif o.type == 4
                    output.Club = [output.Club, num2str(o.value), '  '];
                end
            end
            output.heart = strtrim(output.heart);
            output.Spade = strtrim(output.Spade);
            output.Diamond = strtrim(output.Diamond);
            output.Club = strtrim(output.Club);
            disp(output)
        end


#                 if 1==1
#                     numStr = [num2str(ix),'. Card:'];
#                     disp(numStr)
#                     propList = string(o);
#                     disp(propList)
#                     propgrp = matlab.mixin.util.PropertyGroup(propList);
#                     matlab.mixin.CustomDisplay.displayPropertyGroups(o,propgrp);
#                 end


    end
end





"""