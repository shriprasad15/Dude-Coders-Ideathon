from manim import *
from manim_slides import Slide

class Introduction(Slide):
    def construct(self):
        # --- TITLE SLIDE ---
        title = Text("Solar Panel Detection", gradient=(YELLOW, ORANGE)).scale(1.2)
        subtitle = Text("From Satellite Imagery", font_size=32).next_to(title, DOWN)
        team = Text("Team Dude Coders", font_size=24, color=GRAY).to_corner(DR)
        
        self.play(Write(title), FadeIn(subtitle))
        self.play(Write(team))
        self.next_slide()
        
        self.play(FadeOut(title), FadeOut(subtitle), FadeOut(team))
        
        # --- SCENE 1: THE CHALLENGE ---
        # Visual: 3D Globe/Map simulation
        # Using Grid and Rectangles without LaTeX coordinates
        
        plane = NumberPlane(
            x_range=(-5, 5, 1), 
            y_range=(-5, 5, 1), 
            x_length=10, 
            y_length=10, 
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 2,
                "stroke_opacity": 0.3
            }
        ) # Removed .add_coordinates() to avoid LaTeX
        
        self.play(Create(plane), run_time=2)
        
        # Create "buildings" (Rectangles)
        buildings = VGroup()
        for i in range(5):
            for j in range(4):
                if (i+j) % 2 == 0:
                    rect = Rectangle(height=1.5, width=1.5, fill_color=GREY_E, fill_opacity=1, stroke_color=WHITE)
                    rect.move_to(plane.c2p(i*2 - 4, j*2 - 3))
                    buildings.add(rect)
        
        self.play(FadeIn(buildings))
        
        # Text: The Problem
        problem_text = Text("The Problem: Variable Lighting", font_size=36, color=RED).to_edge(UP)
        self.play(Write(problem_text))
        
        # Simulating Sun movement and Shadows
        # We will change the color of buildings to simulate shadow passing
        sun = Circle(radius=0.5, color=YELLOW, fill_opacity=1).move_to(UP*3 + LEFT*4)
        self.play(FadeIn(sun))
        
        self.next_slide()
        
        # Animate Sun moving
        shadow_overlay = Rectangle(width=12, height=8, color=BLACK, fill_opacity=0).move_to(ORIGIN)
        
        self.play(sun.animate.move_to(UP*3 + RIGHT*4), run_time=3)
        
        # Simulate panels blending in (turn some rectangles dark)
        panels = VGroup()
        for i in range(3):
            p = Rectangle(height=0.4, width=0.8, fill_color=BLUE_E, fill_opacity=0.8, stroke_width=0)
            p.move_to(buildings[i*2].get_center())
            panels.add(p)
            
        self.play(FadeIn(panels))
        self.wait(1)
        
        # Turn everything dark (shadow)
        self.play(
            buildings.animate.set_fill(BLACK), 
            panels.animate.set_fill(BLACK),
            run_time=2
        )
        
        blending_text = Text("Low Contrast = Detection Failure", font_size=32, color=ORANGE).next_to(problem_text, DOWN)
        self.play(Write(blending_text))
        
        self.next_slide()
        self.play(FadeOut(Group(plane, buildings, panels, sun, problem_text, blending_text)))


class FailedExperiments(Slide):
    def construct(self):
        # --- SCENE 2: FAILED EXPERIMENTS ---
        
        title = Text("Attempt 1: Standard Approaches", color=BLUE).to_edge(UP)
        self.play(Write(title))
        
        # Models List
        models = VGroup(
            Text("R-CNN", font_size=36),
            Text("Faster R-CNN", font_size=36),
            Text("Edge Detection", font_size=36)
        ).arrange(DOWN, buff=0.5)
        
        self.play(Write(models))
        self.next_slide()
        
        # Simulate Scanning on an "Image"
        # Create a mock image
        img_rect = Rectangle(width=6, height=4, color=WHITE, fill_color=GREY_D, fill_opacity=1)
        roof = Polygon([-2, -1, 0], [2, -1, 0], [0, 1, 0], color=WHITE, fill_color=GREY_B, fill_opacity=1).move_to(img_rect)
        chimney = Rectangle(width=0.3, height=0.6, fill_color=RED_E, fill_opacity=1).move_to(roof.get_right() + LEFT * 0.5)
        
        mock_scene = Group(img_rect, roof, chimney).scale(0.8).move_to(RIGHT*3)
        
        self.play(models.animate.to_edge(LEFT), FadeIn(mock_scene))
        
        # Scanning effect
        scanner_line = Line(img_rect.get_top(), img_rect.get_bottom(), color=GREEN)
        scanner_line.move_to(img_rect.get_left())
        
        self.play(MoveAlongPath(scanner_line, Line(img_rect.get_left(), img_rect.get_right())), run_time=2)
        self.play(FadeOut(scanner_line))
        
        # False Positives (Red Xs)
        cross1 = Cross(scale_factor=0.2).move_to(chimney)
        cross2 = Cross(scale_factor=0.2).move_to(roof.get_left() + RIGHT*0.5)
        
        self.play(Create(cross1), Create(cross2))
        
        fail_text = Text("False Positives!", color=RED, font_size=40).next_to(mock_scene, DOWN)
        self.play(Write(fail_text))
        
        self.next_slide()
        self.play(FadeOut(Group(title, models, mock_scene, cross1, cross2, fail_text)))


class TheSolution(Slide):
    def construct(self):
        # --- SCENE 3: THE SOLUTION ---
        scale_group = VGroup()
        
        # 1. Class Imbalance
        title = Text("The Breakthrough", color=GREEN).to_edge(UP)
        self.play(Write(title))
        
        balance_text = Text("Addressing Class Imbalance", font_size=32).next_to(title, DOWN)
        self.play(FadeIn(balance_text))
        
        # Scale animation
        scale_base = Triangle().scale(0.5).set_fill(WHITE, opacity=1)
        scale_bar = Line(LEFT*3, RIGHT*3, stroke_width=8)
        scale_bar.next_to(scale_base, UP, buff=0)
        
        left_plate = Circle(radius=0.8, color=WHITE).next_to(scale_bar, UP, aligned_edge=LEFT)
        right_plate = Circle(radius=0.8, color=WHITE).next_to(scale_bar, UP, aligned_edge=RIGHT)
        
        # Load up solar panels
        solar_label = Text("Solar", font_size=20).move_to(left_plate)
        
        self.play(Create(scale_base), Create(scale_bar), Create(left_plate), Create(right_plate))
        self.play(Write(solar_label))
        self.play(Rotate(scale_bar, -0.2, about_point=scale_base.get_top()), run_time=1) # Tip left
        
        # Add Non-Solar data
        non_solar_label = Text("Non-Solar\n(Negative)", font_size=20, color=YELLOW).move_to(right_plate)
        
        self.next_slide()
        self.play(FadeIn(non_solar_label))
        self.play(Rotate(scale_bar, 0.2, about_point=scale_base.get_top()), run_time=1) # Balance
        
        self.play(FadeOut(Group(scale_base, scale_bar, left_plate, right_plate, solar_label, non_solar_label, balance_text)))
        
        # 2. Augmentation & SAM
        aug_text = Text("Augmentation & Precision", font_size=32).next_to(title, DOWN)
        self.play(Write(aug_text))
        
        # Augmentation visual: Square getting brighter
        img_sq = Square(side_length=3, fill_color=BLUE_E, fill_opacity=0.8)
        label_aug = Text("Saturation Boost", font_size=24).next_to(img_sq, DOWN)
        
        self.play(Create(img_sq), Write(label_aug))
        self.play(img_sq.animate.set_fill(color=BLUE_A), run_time=2) # Brighten
        
        # SAM visual: Masking
        mask = Star(n=5, outer_radius=1.2, inner_radius=0.5, color=PINK, fill_opacity=0.5).move_to(img_sq)
        sam_label = Text("SAM Masking", font_size=24, color=PINK).next_to(mask, UP)
        
        self.next_slide()
        self.play(DrawBorderThenFill(mask), Write(sam_label))
        
        self.play(FadeOut(Group(img_sq, label_aug, mask, sam_label, aug_text)))
        
        # 3. YOLOv12 Architecture
        arch_title = Text("YOLOv12 Architecture", font_size=36, color=YELLOW).next_to(title, DOWN)
        self.play(Write(arch_title))
        
        # Simple Block Diagram
        backbone = Rectangle(height=2, width=1.5, fill_color=BLUE, fill_opacity=0.5).to_edge(LEFT, buff=1)
        neck = Rectangle(height=1.5, width=1.5, fill_color=GREEN, fill_opacity=0.5).next_to(backbone, RIGHT, buff=1)
        head = Rectangle(height=1, width=1.5, fill_color=RED, fill_opacity=0.5).next_to(neck, RIGHT, buff=1)
        
        labels = VGroup(
            Text("Backbone", font_size=20).move_to(backbone),
            Text("Neck", font_size=20).move_to(neck),
            Text("Head", font_size=20).move_to(head)
        )
        
        arrows = VGroup(
            Arrow(backbone.get_right(), neck.get_left()),
            Arrow(neck.get_right(), head.get_left())
        )
        
        self.play(Create(backbone), Create(neck), Create(head))
        self.play(Write(labels), Create(arrows))
        
        # Data flow animation
        dot = Dot(color=YELLOW).move_to(backbone.get_left())
        self.play(MoveAlongPath(dot, Line(backbone.get_left(), head.get_right())), run_time=2)
        
        self.next_slide()
        self.play(FadeOut(Group(title, arch_title, backbone, neck, head, labels, arrows, dot)))


class TrainingOps(Slide):
    def construct(self):
        # --- SCENE 4: Training Optimizations ---
        title = Text("Training Optimization", color=PURPLE).to_edge(UP)
        gpu = Rectangle(width=2, height=2, color=GREEN).set_fill(BLACK, opacity=1)
        gpu_label = Text("V100 GPU", color=GREEN, font_size=24).move_to(gpu)
        
        self.play(Write(title), Create(gpu), Write(gpu_label))
        
        # List of optimizations
        # BulletedList uses LaTeX, so we construct manually
        opts = VGroup()
        points = [
            "Mixed Precision (FP16)",
            "Cosine Annealing",
            "Mosaic Augmentation"
        ]
        
        for p_str in points:
            row = VGroup(
                Dot(color=WHITE),
                Text(p_str, font_size=28)
            ).arrange(RIGHT, buff=0.2)
            opts.add(row)
            
        opts.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        opts.next_to(gpu, RIGHT, buff=1)
        
        self.play(Write(opts))
        
        self.next_slide()
        self.play(FadeOut(Group(title, gpu, gpu_label, opts)))


class TheInferenceCascade(Slide):
    def construct(self):
        # --- SCENE 5: The Inference Strategy ---
        title = Text("The Secret Sauce: Inference Cascade", color=GOLD).to_edge(UP)
        self.play(Write(title))
        
        # Flowchart boxes
        step1 = Rectangle(width=3, height=1, color=BLUE).shift(UP*2)
        txt1 = Text("1. Buffer 1200", font_size=24).move_to(step1)
        
        step2 = Rectangle(width=3, height=1, color=ORANGE).shift(RIGHT*4)
        txt2 = Text("2. Saturation Boost", font_size=24).move_to(step2)
        
        step3 = Rectangle(width=3, height=1, color=RED).shift(DOWN*2)
        txt3 = Text("3. Crop & Zoom", font_size=24).move_to(step3)
        
        step4 = Rectangle(width=3, height=1, color=PURPLE).shift(LEFT*4)
        txt4 = Text("4. Buffer 2400", font_size=24).move_to(step4)
        
        # Connect them
        arrow1 = Arrow(step1.get_right(), step2.get_left()) # 1 -> 2
        arrow2 = Arrow(step2.get_bottom(), step3.get_top()) # 2 -> 3
        arrow3 = Arrow(step3.get_left(), step4.get_right()) # 3 -> 4
        
        # Found box
        found_box = Circle(radius=0.8, color=GREEN, fill_opacity=0.5).move_to(ORIGIN)
        found_txt = Text("FOUND!", font_size=24).move_to(found_box)
        
        # All arrows to found
        to_found1 = Arrow(step1.get_bottom(), found_box.get_top(), color=GREEN)
        to_found2 = Arrow(step2.get_bottom(), found_box.get_top(), color=GREEN)
        to_found3 = Arrow(step3.get_top(), found_box.get_bottom(), color=GREEN) # Fix direction visually
        to_found4 = Arrow(step4.get_right(), found_box.get_left(), color=GREEN)
        
        self.play(Create(step1), Write(txt1))
        self.next_slide()
        
        # Simulate logic flow
        # Try 1 -> Fail
        start_dot = Dot(color=YELLOW).move_to(step1.get_left())
        self.play(FadeIn(start_dot))
        self.play(start_dot.animate.move_to(step1.get_right()))
        
        self.play(Create(arrow1))
        self.play(Create(step2), Write(txt2))
        self.play(start_dot.animate.move_to(step2.get_bottom()))
        
        self.next_slide()
        self.play(Create(arrow2))
        self.play(Create(step3), Write(txt3))
        self.play(start_dot.animate.move_to(step3.get_left()))
        
        self.next_slide()
        self.play(Create(arrow3))
        self.play(Create(step4), Write(txt4))
        
        # Show success case
        self.play(FadeIn(found_box), Write(found_txt))
        self.play(Create(to_found4)) # Success at step 4
        self.play(Indicate(found_box))
        
        self.next_slide()
        self.play(FadeOut(Group(title, step1, txt1, step2, txt2, step3, txt3, step4, txt4, 
                                arrow1, arrow2, arrow3, found_box, found_txt, to_found1, to_found2, to_found3, to_found4, start_dot)))


class Benchmarks(Slide):
    def construct(self):
        # --- SCENE 6: Benchmarks ---
        title = Text("Benchmarks & Comparison", color=TEAL).to_edge(UP)
        self.play(Write(title))
        
        # Manual Bar Chart Construction (Avoiding LaTeX)
        axes = VGroup(
            Line(start=DOWN*2+LEFT*4, end=DOWN*2+RIGHT*4), # X axis
            Line(start=DOWN*2+LEFT*4, end=UP*2+LEFT*4)    # Y axis
        )
        
        # Data: R-CNN (.65), Faster-RCNN (.78), YOLOv8 (.85), Ours (.92)
        # Max height ~ 4 units
        
        bars = VGroup()
        names = VGroup()
        
        data = [
            ("R-CNN", 0.65, GREY),
            ("Faster\nR-CNN", 0.78, TEAL),
            ("YOLOv8", 0.85, BLUE),
            ("Ours", 0.92, GOLD)
        ]
        
        for i, (name, val, col) in enumerate(data):
            bar_height = val * 4
            bar = Rectangle(width=1, height=bar_height, fill_color=col, fill_opacity=0.8)
            bar.move_to(DOWN*2 + LEFT*2.5 + RIGHT*(i*1.8), aligned_edge=DOWN)
            
            label_name = Text(name, font_size=18).next_to(bar, DOWN)
            label_val = Text(f"{val:.2f}", font_size=18).next_to(bar, UP)
            
            bars.add(bar)
            names.add(label_name, label_val)
            
        self.play(Create(axes))
        self.play(GrowFromEdge(bars, DOWN))
        self.play(FadeIn(names))
        
        # Emphasize Ours
        self.play(Indicate(bars[3]))
        
        score_text = Text("Top Performance!", color=GOLD, font_size=36).to_corner(UR)
        self.play(Write(score_text))
        
        self.next_slide()
        self.play(FadeOut(Group(title, axes, bars, names, score_text)))

class Conclusion(Slide):
    def construct(self):
        # --- SCENE 7: CONCLUSION ---
        text = Text("Precision through Persistence", gradient=(YELLOW, RED), font_size=48)
        self.play(Write(text))
        self.wait(2)
