function visualize(number)

    %convert to black and white
    map = [1 1 1
        %possible to add more colors.
        0 0 0
    ];
    colormap(map)
    
    %display the number
    imagesc(number);
end