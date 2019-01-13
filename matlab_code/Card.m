classdef Card < matlab.mixin.CustomDisplay
    % Card Summary of this class goes here
    % this is the first line 
    % Detailed explanation goes here
    % CARD 123
    
    properties
        % Value is 1->A, 2->2, ... 10->10, 11->Jack, 12->Queen, 13-> King
        value
        % Type is 1->Heart, 2->Spade, 3->Diamond, 4->Clover
        type
        % point is A->4, King->3, Queen->2, Jack->1
        point
        % representation of this object
        repr
    end
    
    methods
        function obj = Card(index1)
            %CARD Construct an instance of this class
            %   Detailed explanation goes here
            obj.value = mod(index1, 13) + 1;

            obj.type = idivide(int32(index1),13) + 1;
            if obj.value == 1
                obj.point = 4;
            elseif obj.value == 13
                obj.point = 3;
            elseif obj.value == 12
                obj.point = 2;
            elseif obj.value == 11
                obj.point = 1;
            else
                obj.point = 0;
            end
             if obj.type == 1
                result = 'Heart ';
            elseif obj.type == 2
                result = 'Spade ';
            elseif obj.type == 3
                result = 'Diamond ';
            elseif obj.type == 4
                result = 'Clover ';
            end
            
            result = strcat(result, string(obj.value));
            obj.repr = result;

        end
        
        

        
        function outputArg = method1(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
        
        function show(obj)
            if obj.type == 1
                result = 'Heart ';
            elseif obj.type == 2
                result = 'Spade ';
            elseif obj.type == 3
                result = 'Diamond ';
            elseif obj.type == 4
                result = 'Clover ';
            end
            
            result = strcat(result, string(obj.value));
            % disp(result)
            
        end
        
        function output = string(obj)
            output = obj.repr;
        end
            
    end
    
    methods (Access = protected)

        function displayScalarObject(obj)
            % Implement the custom display for scalar obj
            disp(obj.repr)
        end
        
        function displayNonScalarObject(objAry)
            dimStr = matlab.mixin.CustomDisplay.convertDimensionsToString(objAry);
            cName = matlab.mixin.CustomDisplay.getClassNameForHeader(objAry);
            headerStr = [dimStr,' ',cName,' members:'];
            header = sprintf('%s\n',headerStr);
            disp(header);
%             disp(find([objAry.type] == 1));
%             index = [objAry.type] == 1;
%             disp(objAry(index).value);
%             heart = table.hand1( find( [table.hand1.type]==1) );
%             disp(heart)
            output.heart = '';
            output.Spade = '';
            output.Diamond = '';
            output.Clover = '';
            for ix = 1:length(objAry)
                o = objAry(ix);
                if o.type == 1
                    output.heart = [output.heart, num2str(o.value), '  '];
                elseif o.type == 2
                    output.Spade = [output.Spade, num2str(o.value), '  '];
                elseif o.type == 3
                    output.Diamond = [output.Diamond, num2str(o.value), '  '];
                elseif o.type == 4
                    output.Clover = [output.Clover, num2str(o.value), '  '];
                end
            end
            output.heart = strtrim(output.heart);
            output.Spade = strtrim(output.Spade);
            output.Diamond = strtrim(output.Diamond);
            output.Clover = strtrim(output.Clover);
            disp(output)
        end


%                 if 1==1
%                     numStr = [num2str(ix),'. Card:'];
%                     disp(numStr)
%                     propList = string(o);
%                     disp(propList)
%                     propgrp = matlab.mixin.util.PropertyGroup(propList);
%                     matlab.mixin.CustomDisplay.displayPropertyGroups(o,propgrp);
%                 end

            
    end
end





