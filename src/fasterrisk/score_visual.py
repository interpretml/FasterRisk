import os
import re
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

current_dir = os.path.dirname(os.path.abspath(__file__))
font_map = {
    'Noto': os.path.join(current_dir, 'fonts/NotoSansMath-Regular.ttf'),
    }

def save_img_to_pdf(img, path: str):
    img.save(path, "PDF", resolution=100.0)

def find_crop_x_boundaries(img):
    width, height = img.size
    pixels = img.load()

    white_color = (255, 255, 255)
    
    leftmost_x = None
    rightmost_x = None
    
    for x in range(width):
        all_white = True

        for y in range(height):
            if pixels[x, y] != white_color:
                all_white = False
                break
        if not all_white and leftmost_x is None:
            leftmost_x = x 
        elif all_white and leftmost_x is not None and rightmost_x is None:
            rightmost_x = x-1

    return leftmost_x, rightmost_x

def combine_images(bottom_image, top_image, top_image_y_offset=0, bottom_image_y_offset=-90, crop_margin=40):
    
    width = max(bottom_image.width, top_image.width)
    height = bottom_image.height + top_image.height + top_image_y_offset + bottom_image_y_offset
    combined_image = Image.new('RGB', (width, height), color=(255, 255, 255))

    combined_image.paste(top_image, (int((width - top_image.width)/2), top_image_y_offset))
    combined_image.paste(bottom_image, (int((width - bottom_image.width)/2), top_image.height + top_image_y_offset + bottom_image_y_offset))
    
    return combined_image
    # if sharpen != 1:
    #     # Create a Sharpness object and set the sharpness factor
    #     sharpness = ImageEnhance.Sharpness(combined_image)
    #     # Apply the sharpness enhancement
    #     combined_image = sharpness.enhance(sharpen)
        
    # left_most, right_most = find_crop_x_boundaries(combined_image)
    
    # delta_most = max(width/2 - left_most, right_most - width/2)
    # allowance = width/2 - delta_most
    # crop_margin = min(crop_margin, allowance)
    
    # left_crop = allowance - crop_margin
    # right_crop = width - (allowance - crop_margin)
    
    # return combined_image.crop((left_crop,0, right_crop, height))

def output_to_score_risk_df(input_text: str):
    lines = input_text.split('\n')
    
    score_lines = [line for line in lines if line.startswith('SCORE')]
    risk_lines = [line for line in lines if line.startswith('RISK')]
    
    score_tokens = [token.strip() for score_line in score_lines for token in score_line.split('|')[1:] if token.strip() != '']
    risk_tokens = [token.strip() for risk_line in risk_lines for token in risk_line.split('|')[1:] if token.strip() != '']
    
    scores = [float(token) for token in score_tokens]
    risks = [float(token[:-1]) for token in risk_tokens]
    
    return pd.DataFrame({'Score': scores, 'Risk': risks})  
    
def scrape_to_df(input_text: str):
    pattern = r'(\d+\.)\s*([^<]+)<=([-\d.]+)\s*(-?\d+)'
    Line = namedtuple("Line", ["feature", "cutoff", "points"])
    lines = []

    for line in input_text.split('\n'):
        match = re.match(pattern, line)
        if match:
            lines.append(Line(match.group(2), match.group(3), match.group(4)))
        else:
            pass
            # print("No match found in this line:", line)

    # Create a DataFrame using the extracted information
    df = pd.DataFrame(data=lines)
    df.cutoff = df.cutoff.astype(float)
    df.points = df.points.astype(float)
    return df

def compute_cumulative(df: pd.DataFrame):
    df_sorted = df.sort_values(by=['feature', 'cutoff'], ascending=True)
    df_sorted['running_sum'] = df_sorted.groupby('feature')['points'].transform(lambda x: x[::-1].cumsum()[::-1])
    df_interval = df_sorted[['feature', 'cutoff', 'running_sum']]
    df_interval.columns = ['feature', 'interval_upper', 'interval_points']
    df_interval['interval_lower'] = df_interval.groupby('feature')['interval_upper'].shift(1)
    df_interval['interval_lower'].fillna(float('-inf'), inplace=True)
    new_rows = []
    for feature, group in df_interval.groupby('feature'):
        max_interval_upper = group['interval_upper'].max()
        new_rows.append({'feature': feature, 'interval_lower': max_interval_upper, 'interval_upper': np.inf, 'interval_points': 0})

    new_df = pd.DataFrame(new_rows)
    result = pd.concat([df_interval, new_df], axis=0, ignore_index=True).sort_values(['feature', 'interval_lower'])
    return result[['feature', 'interval_lower', 'interval_upper', 'interval_points']]
    
def compute_offset(df_interval: pd.DataFrame):
    needed_offset = np.sum(df_interval.groupby('feature').min()['interval_points'])
    df_offset = df_interval.copy()
    df_offset['interval_points'] -= df_offset.groupby('feature')['interval_points'].transform(lambda x:x.min())
    return df_offset, needed_offset

def output_to_score_intervals(input_text: str):
    return compute_offset(compute_cumulative(scrape_to_df(input_text)))

class ScoreCardVisualizer:
    
    def __init__(self, df: pd.DataFrame, needed_offset: int):
        self.df = df
        assert(df.columns.tolist() == ['feature', 'interval_lower', 'interval_upper', 'interval_points'])
        self.needed_offset = needed_offset
        self.compute_priority()
        # self.compute_offset()
        
    def compute_priority(self):
        self.unique_features = self.df.groupby('feature')[['interval_points']].max().reset_index().sort_values(by='interval_points', ascending=False)['feature']
        
    def get_dynamic_height(self):
        return len(self.unique_features) * self.box_height + self.vertical_padding
    
    def get_dynamic_width(self):
        return self.df.groupby('feature').count()['interval_points'].max() * 2 * self.box_width + self.horizontal_padding
        
    def set_visual_sizes(self, 
                          vertical_padding: int = 140,
                          horizontal_padding: int = 0,
                          box_height: int = 90,
                          box_width: int = 300, 
                          center_box_width: int = 600,
                          border_width: int = 1,
                          **kwargs
                          ):
        self.vertical_padding = vertical_padding
        self.horizontal_padding = horizontal_padding
        self.box_height = box_height
        self.box_width = box_width
        self.center_box_width = center_box_width
        self.border_width = border_width
        
        self.img_width = self.get_dynamic_width()
        self.img_height = self.get_dynamic_height()
        
    def set_fonts(self, 
                  font_dir: str = font_map['Noto'],
                  title_font_size: int = 40,
                  bigger_font_size: int = 32,
                  smaller_font_size: int = 32,
                  **kwargs):
        
        self.font_dir = font_dir
        
        self.title_font_size = title_font_size
        self.bigger_font_size = bigger_font_size
        self.smaller_font_size = smaller_font_size
        
        self.title_font = ImageFont.truetype(font=self.font_dir, size=self.title_font_size)
        self.bigger_font = ImageFont.truetype(font=self.font_dir, size=self.bigger_font_size)
        self.smaller_font = ImageFont.truetype(font=self.font_dir, size=self.smaller_font_size)

    def generate_visual_card(self, 
                             title: str, 
                             custom_row_order: list[str] = None,
                             smart_detect_binary: bool = True,
                             title_y_offset: int = -20,
                             **kwargs):
        # draw canvas
        if custom_row_order is not None:
            if set(custom_row_order) != set(self.unique_features):
                print(f"df contains features {set(self.unique_features) - set(custom_row_order)} which passed custom row order does not")
                print(f"custom row order contains {set(custom_row_order)-set(self.unique_features)} which df does not")
                raise ValueError("The custom row ordering must contain the same features as the df passed")
            self.row_ordering = custom_row_order
        else:
            self.row_ordering = self.unique_features
        
        self.title = title
        self.smart_detect_binary = smart_detect_binary
        self.current_boolean_feature = False
        
        self.set_fonts(**kwargs)
        self.set_visual_sizes(**kwargs)

        self.make_frame()
        self.draw_title(title, title_y_offset)
        
        # draw rows - loop through unique features
        self.draw_rows()
        
        # self.draw_footer()

        return self.img
    
    def draw_title(self, text, title_y_offset):
        text_x = self.img_width / 2 - self.title_font.getsize(text)[0] / 2
        text_y = self.vertical_padding / 4 - self.title_font.getsize(text)[1] / 2 + title_y_offset
        self.draw.text((text_x, text_y), text, font=self.title_font, fill="black")
        
    def draw_footer(self, text=None):
        if text is None:
            text = f"Finally, add {int(self.needed_offset)} to score."
        text_x = self.img_width / 2 - self.bigger_font.getsize(text)[0] / 2
        text_y = (self.img_height - self.vertical_padding / 4) - self.bigger_font.getsize(text)[1] / 2
        self.draw.text((text_x, text_y), text, font=self.bigger_font, fill="black")
        
    def make_frame(self):
        self.img = Image.new('RGB', (self.img_width, self.img_height), color='white')
        self.draw = ImageDraw.Draw(self.img)
        
    def draw_rows(self):
        for row_i, feature in enumerate(self.row_ordering):
            feature_df = self.df[self.df['feature'] == feature]
            self.draw_row(feature_df, row_i)

    def draw_row(self, feature_df, row_num):
        # dummy
        feature_df = feature_df.copy().reset_index(drop=True)
        
        reference_row = feature_df[feature_df["interval_points"] == 0].index[0]
        
        for index, row in feature_df.iterrows():
            if self.smart_detect_binary and len(feature_df) == 2 and feature_df['interval_upper'].iloc[0] == 0:
                self.current_boolean_feature = True
            else:
                self.current_boolean_feature = False
            upper_left = self.get_upper_left(row_num, index - reference_row)
            lower_right = self.get_lower_right(row_num, index - reference_row)
            text = self.interval_to_string(row, True if index == reference_row else False)
            self.draw_box_with_text(upper_left, lower_right, text, (200, 200, 200) if row_num%2==0 else "white")

    def get_upper_left(self, row_num: int, column_offset: int):
        x_middle = self.img_width / 2
        
        y = self.vertical_padding/2 + row_num * self.box_height
        x = x_middle + (np.clip(column_offset, 0, 1) - 0.5) * self.center_box_width + (column_offset - np.clip(column_offset, 0, 1)) * self.box_width
        return (x,y)

    def get_lower_right(self, row_num: int, column_offset: int):
        upper_left = self.get_upper_left(row_num, column_offset)
        x, y = upper_left
        if column_offset == 0: 
            return (x + self.center_box_width, y + self.box_height)
        else:
            return (x + self.box_width, y + self.box_height)

    def draw_box_with_text(self, upper_left, lower_right, text: str, box_shade):
        self.draw_box(upper_left, lower_right, box_shade)
        self.draw_centered_text(upper_left, lower_right, text)
    
    def draw_box(self, upper_left, lower_right, box_shade):
        x, y = upper_left
        xp, yp = lower_right
        self.draw.rectangle([x, y, xp, yp], outline="black", fill=box_shade, width=self.border_width)
        
    def calc_text_height(self, text):
        # assumes 1st line of text is large, 2nd line small
        
        return self.bigger_font.getsize("hg")[1] + (len(text.split("\n"))-1) * self.smaller_font.getsize("hg")[1]

    def draw_centered_text(self, upper_left, lower_right, text: str): 
        lines = text.split('\n')
        
        total_height = self.calc_text_height(text)  # total height of all lines of text
        
        x, y = upper_left
        xp, yp = lower_right
        
        top = y + ((yp - y) - total_height) / 2  # top position of the first line
        
        # draw each line centered within the rectangle
        for i, line in enumerate(lines):
            text_y = top
            cur_font = self.bigger_font if i == 0 else self.smaller_font
            line_width = cur_font.getsize(line)[0]
            line_height = cur_font.getsize(line)[1]
            text_x = x + (xp - x) / 2 - line_width / 2
            self.draw.text((text_x, text_y), line, font=cur_font, fill="black")
            top += line_height

    def interval_to_string(self, df_row, include_feature):
        result = ""
        
        # name
        if include_feature: 
            if not self.current_boolean_feature:
                if df_row['interval_lower'] == float('-inf'):
                    result += f"{df_row['feature']} \u2264 {df_row['interval_upper']:.2f}"
                elif df_row['interval_upper'] == float('inf'):
                    result += f"{df_row['interval_lower']:.2f} < {df_row['feature']}"
                else:
                    result += f"{df_row['interval_lower']:.2f} < {df_row['feature']} \u2264 {df_row['interval_upper']:.2f} "
            elif self.current_boolean_feature:
                if df_row['interval_upper'] == 0:
                    result += f"{df_row['feature']} = NO"
                else:
                    result += f"{df_row['feature']} = YES"

        
        # numeric interval
        else:
            if not self.current_boolean_feature:
                if df_row['interval_lower'] == float('-inf'):
                    result += f"\u2264 {df_row['interval_upper']:.2f}"
                elif df_row['interval_upper'] == float('inf'):
                    result += f"{df_row['interval_lower']:.2f} <"
                else:
                    result += f"{df_row['interval_lower']:.2f} - {df_row['interval_upper']:.2f} "
            elif self.current_boolean_feature:
                if df_row['interval_upper'] == 0:
                    result += "NO"
                else:
                    result += "YES"
                
        result += "\n"

        result += f"{int(df_row['interval_points'])}"
            
        return result
        # points

class TableVisualizer:
    def __init__(self, df_table, offset):
        self.df_table = df_table
        self.df_table['Score'] -= offset
        
    def setup(self, **kwargs):
        self.set_visual_sizes(**kwargs)
        self.set_fonts(**kwargs)
        
    def set_visual_sizes(self, 
                        vertical_padding: int = 250,
                        horizontal_padding: int = 42,
                        box_height: int = 50,
                        box_width: int = 100, 
                        border_width: int = 1,
                        **kwargs
                        ):
        self.vertical_padding = vertical_padding
        self.horizontal_padding = horizontal_padding
        self.box_height = box_height
        self.box_width = box_width
        self.border_width = border_width
        
        self.img_width = self.get_dynamic_width()
        self.img_height = self.get_dynamic_height()
        
    def set_fonts(self, 
                  font_dir: str = font_map['Noto'],
                  title_font_size: int = 60,
                  bigger_font_size: int = 28,
                  smaller_font_size: int = 20,
                  **kwargs):
        
        self.font_dir = font_dir
        
        self.title_font_size = title_font_size
        self.bigger_font_size = bigger_font_size
        self.smaller_font_size = smaller_font_size
        
        self.title_font = ImageFont.truetype(font=self.font_dir, size=self.title_font_size)
        self.bigger_font = ImageFont.truetype(font=self.font_dir, size=self.bigger_font_size)
        self.smaller_font = ImageFont.truetype(font=self.font_dir, size=self.smaller_font_size)

    def make_frame(self):
        self.img = Image.new('RGB', (self.img_width, self.img_height), color='white')
        self.draw = ImageDraw.Draw(self.img)
        
    def generate_table(self, title, **kwargs):
        
        self.setup(**kwargs)
        self.make_frame()
        
        self.draw_title(title)
        self.draw_table_header()
        
        for idx, row in self.df_table.iterrows():
            self.draw_row(idx, row)
        
        return self.img
    
    def draw_table_header(self):
        self.draw_box_with_text(self.get_upper_left(-1, 0), self.get_lower_right(-1, 0), (255,255,255), "Score")
        self.draw_box_with_text(self.get_upper_left(-1, 1), self.get_lower_right(-1, 1), (200,200,200), "Risk")
        
    def get_dynamic_width(self):
        return (len(self.df_table) + 1) * self.box_width + self.horizontal_padding
    
    def get_dynamic_height(self):
        return self.box_height * 2 + self.vertical_padding
    
    def draw_row(self, row, df_row):
        self.draw_box_with_text(self.get_upper_left(row, 0), self.get_lower_right(row, 0), (255,255,255), f"{int(df_row['Score'])}")
        self.draw_box_with_text(self.get_upper_left(row, 1), self.get_lower_right(row, 1), (200,200,200), f"{df_row['Risk']}%")
        
    def draw_centered_text(self, upper_left, lower_right, text):
        line_width, line_height = self.bigger_font.getsize(text)
        x, y = upper_left
        xp, yp = lower_right
        text_x = x + (xp - x) / 2 - line_width / 2
        text_y = y + (yp - y) / 2 - line_height / 2
        self.draw.text((text_x, text_y), text, font=self.bigger_font, fill="black")
        
    def draw_box(self, upper_left, lower_right, background):
        x, y = upper_left
        xp, yp = lower_right
        self.draw.rectangle([x, y, xp, yp], outline="black", fill=background, width=self.border_width)
        
    def get_upper_left(self, row, col):
        return (self.horizontal_padding/2 + (row+1) * self.box_width, self.img_height/2 + (col-1) * self.box_height)

    def get_lower_right(self, row, col):
        x,y = self.get_upper_left(row, col)
        return (x + self.box_width, y + self.box_height)
    
    def draw_box_with_text(self, upper_left, lower_right, background, text):
        self.draw_box(upper_left, lower_right, background)
        self.draw_centered_text(upper_left, lower_right, text)
        
    def draw_title(self, title):
        line_width, line_height = self.title_font.getsize(title)
        x = self.img_width / 2 - line_width / 2
        y = self.vertical_padding/4 - line_height/2
        self.draw.text((x, y), title, font=self.title_font, fill="black")
        
    def generate_plot(self, title):
        plt.plot(self.df_table.Score, self.df_table.Risk)
        plt.xlabel("Score")
        plt.ylabel("Risk (%)")
        plt.title(title)
        plt.show()