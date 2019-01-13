classdef Table 
    %TABLE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        hand1 = []
        hand2 = []
        hand3 = []
        hand4 = []
        hand ;
    end
    
    methods
        function obj = Table(rdn_list)
            %TABLE Construct an instance of this class
            %   Detailed explanation goes here
           
            sorted_list = sort(rdn_list(:));
            rdn_list = reshape(rdn_list, [4, 13]);
            obj.hand = CardArray(rdn_list);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%% New constructor %%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             for i = 1:4
%                 for j = 1:13
%                     obj.hand(i,j) = Card(rdn_list(i,j)) ;
%                 end
%             end
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%% old constructor %%%%%%%%%%%%%%%%%%
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             for i = 1:4
%                 one_hand_list = rdn_list(i,:);
%                 for j = 1:13
%                     % disp(one_hand_list)
%                     index = find(one_hand_list(j)==sorted_list)-1;
%                     % disp(index)
%                     hand_num = ['obj.hand', num2str(i)];
%                     string1 = [hand_num,'=[', hand_num, ',Card(', char(string(index)), ')];' ];
%                     % disp(string1);
%                     eval(string1);                     
%                     % disp(obj.hand1)
%                 end
%             end
                    
        end
        
        function outputArg = method1(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
        
        function result = calculate_score(obj)
            result = 0;
            for i = 1:13
                result = result + obj.hand(1,i).Value.point;
            end
            
        end

    end
end

